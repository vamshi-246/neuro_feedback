// Linear layer + argmax: logit_k = bias_k + sum_i W[k][i] * q[i], k = 0..2.
//
// W and bias are the folded, 8-bit-input weights (Wq, bq) scaled by 2^24 and
// rounded (signed 28 bit). q[i] are the unsigned 8-bit feature codes. No softmax:
// it is monotonic, so argmax over the logits is the same class.
// Class 0 = Low (<4), 1 = Moderate (4..<7), 2 = High (>=7). Ties go to the lower index.
module linear_classifier #(
    parameter W_FILE = "weights.mem",
    parameter B_FILE = "bias.mem"
) (
    input  wire               clk,
    input  wire               rst,
    input  wire               start,
    input  wire [127:0]       q_flat,
    output reg  signed [39:0] logit0,
    output reg  signed [39:0] logit1,
    output reg  signed [39:0] logit2,
    output reg  [1:0]         cls,
    output reg                done
);
    reg  [5:0]  waddr;
    wire [27:0] wdata;
    rom_sync #(.W(28), .AW(6), .FILE(W_FILE)) u_w (.clk(clk), .addr(waddr), .data(wdata));

    reg  [1:0]  baddr;
    wire [27:0] bdata;
    rom_sync #(.W(28), .AW(2), .FILE(B_FILE)) u_b (.clk(clk), .addr(baddr), .data(bdata));

    localparam S_IDLE = 3'd0, S_BIAS = 3'd1, S_BW = 3'd2, S_BG = 3'd3,
               S_WA = 3'd4, S_WW = 3'd5, S_WG = 3'd6, S_ARG = 3'd7;
    reg [2:0] state;
    reg [1:0] k;
    reg [3:0] i;
    reg signed [39:0] acc;

    wire [7:0]         qi   = q_flat[8 * i +: 8];
    wire signed [36:0] prod = $signed(wdata) * $signed({1'b0, qi});

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            state <= S_IDLE;
        end else begin
            case (state)
            S_IDLE: if (start) begin
                k     <= 2'd0;
                state <= S_BIAS;
            end
            S_BIAS: begin baddr <= k; state <= S_BW; end
            S_BW:   state <= S_BG;
            S_BG: begin
                acc   <= {{12{bdata[27]}}, bdata};
                i     <= 4'd0;
                state <= S_WA;
            end
            S_WA: begin waddr <= {k, i}; state <= S_WW; end
            S_WW: state <= S_WG;
            S_WG: begin
                if (i == 4'd15) begin
                    case (k)
                    2'd0: logit0 <= acc + prod;
                    2'd1: logit1 <= acc + prod;
                    default: logit2 <= acc + prod;
                    endcase
                    if (k == 2'd2) state <= S_ARG;
                    else begin k <= k + 2'd1; state <= S_BIAS; end
                end else begin
                    acc   <= acc + prod;
                    i     <= i + 4'd1;
                    state <= S_WA;
                end
            end
            S_ARG: begin
                if (logit1 > logit0)
                    cls <= (logit2 > logit1) ? 2'd2 : 2'd1;
                else
                    cls <= (logit2 > logit0) ? 2'd2 : 2'd0;
                done  <= 1'b1;
                state <= S_IDLE;
            end
            endcase
        end
    end
endmodule

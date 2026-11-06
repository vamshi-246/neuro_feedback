// 8-bit feature quantizer by binary threshold search.
//
// Training quantizes each feature f as q = round(clip((f - lo) / (hi - lo), 0, 1) * 255).
// That equals the number of thresholds T_k = lo + (k - 0.5)(hi - lo)/255, k = 1..255,
// that f reaches. Every feature here is a monotone function of a ratio A / B with
// B > 0, so "f >= T_k" becomes the exact integer test
//
//        A * 2^40  >=  C_k * B,        C_k = round(c(T_k) * 2^40)  (ROM)
//
// where c() is T, e^T or T^2 depending on the feature (hw_golden.FORMS). This avoids
// log, division and square root in hardware, and reproduces the software codes
// except where f lies within rounding distance of a threshold.
//
// Binary search: 8 compares per feature, 16 features, ~80 cycles per compare.
module threshold_quantizer #(
    parameter TH_FILE = "thresh.mem"
) (
    input  wire               clk,
    input  wire               rst,
    input  wire               start,
    output reg  [3:0]         sel,          // feature index -> feature_engine
    input  wire signed [71:0] a_in,
    input  wire signed [71:0] b_in,
    output reg  [127:0]       q_flat,       // q[i] = q_flat[8*i +: 8]
    output reg                done
);
    reg  [11:0] rom_addr;
    wire [55:0] rom_data;
    rom_sync #(.W(56), .AW(12), .FILE(TH_FILE)) u_th (
        .clk(clk), .addr(rom_addr), .data(rom_data));

    reg                 mul_start;
    wire                mul_done;
    wire signed [127:0] mul_p;
    mul_seq #(.WA(56), .WB(72)) u_mul (
        .clk(clk), .rst(rst), .start(mul_start), .a($signed(rom_data)), .b(b_in),
        .busy(), .done(mul_done), .p(mul_p));

    localparam S_IDLE = 3'd0, S_ADDR = 3'd1, S_RW1 = 3'd2, S_RW2 = 3'd3,
               S_MUL = 3'd4, S_CMP = 3'd5, S_DONE = 3'd6;
    reg [2:0] state;
    reg [2:0] bitn;
    reg [7:0] code;
    wire [7:0] k = code | (8'd1 << bitn);

    wire signed [129:0] lhs  = {{18{a_in[71]}}, a_in, 40'd0};
    wire signed [129:0] rhs  = {{2{mul_p[127]}}, mul_p};
    wire                pass = (lhs >= rhs);
    wire [7:0]          code_next = pass ? k : code;

    always @(posedge clk) begin
        done      <= 1'b0;
        mul_start <= 1'b0;
        if (rst) begin
            state <= S_IDLE;
        end else begin
            case (state)
            S_IDLE: if (start) begin
                sel   <= 4'd0;
                code  <= 8'd0;
                bitn  <= 3'd7;
                state <= S_ADDR;
            end
            S_ADDR: begin
                rom_addr <= {sel, k};
                state    <= S_RW1;
            end
            S_RW1: state <= S_RW2;
            S_RW2: begin                    // rom_data valid; b_in stable (sel fixed)
                mul_start <= 1'b1;
                state     <= S_MUL;
            end
            S_MUL: if (mul_done) state <= S_CMP;
            S_CMP: begin
                if (bitn == 3'd0) begin
                    q_flat[8 * sel +: 8] <= code_next;
                    code <= 8'd0;
                    bitn <= 3'd7;
                    if (sel == 4'd15) begin
                        state <= S_DONE;
                    end else begin
                        sel   <= sel + 4'd1;
                        state <= S_ADDR;
                    end
                end else begin
                    code  <= code_next;
                    bitn  <= bitn - 3'd1;
                    state <= S_ADDR;
                end
            end
            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule

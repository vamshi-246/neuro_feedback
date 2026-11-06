// Sequential signed multiplier (shift-and-add on magnitudes), WB+1 cycles per product.
// Used for the low-rate wide arithmetic (feature finalisation, quantizer compares),
// where width matters and speed does not: it costs adders and registers, no DSPs.
//
// Pulse `start` with a and b valid; `done` pulses one cycle with p = a * b (exact).
module mul_seq #(
    parameter WA = 72,
    parameter WB = 72
) (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     start,
    input  wire signed [WA-1:0]     a,
    input  wire signed [WB-1:0]     b,
    output reg                      busy,
    output reg                      done,
    output reg  signed [WA+WB-1:0]  p
);
    // Magnitudes. For the most negative value the WA-bit negation wraps to
    // 2^(WA-1), which is the correct magnitude when read as unsigned.
    wire [WA-1:0] abs_a = a[WA-1] ? (~a + 1'b1) : a;
    wire [WB-1:0] abs_b = b[WB-1] ? (~b + 1'b1) : b;

    reg [WA+WB-1:0] acc;
    reg [WA+WB-1:0] mcand;
    reg [WB-1:0]    mplier;
    reg             neg;
    reg [7:0]       cnt;

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            busy <= 1'b0;
            p    <= {(WA+WB){1'b0}};
        end else if (start) begin
            acc    <= {(WA+WB){1'b0}};
            mcand  <= {{WB{1'b0}}, abs_a};
            mplier <= abs_b;
            neg    <= a[WA-1] ^ b[WB-1];
            cnt    <= WB;
            busy   <= 1'b1;
        end else if (busy) begin
            if (cnt == 0) begin
                p    <= neg ? -$signed(acc) : $signed(acc);
                busy <= 1'b0;
                done <= 1'b1;
            end else begin
                if (mplier[0])
                    acc <= acc + mcand;
                mcand  <= mcand << 1;
                mplier <= mplier >> 1;
                cnt    <= cnt - 1'b1;
            end
        end
    end
endmodule

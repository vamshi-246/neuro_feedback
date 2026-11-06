// Sequential integer square root: root = floor(sqrt(n)), n unsigned 2*WR bits.
// Bitwise restoring algorithm, WR iterations (+1 to finish).
module isqrt_seq #(
    parameter WR = 72
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    input  wire [2*WR-1:0]   n,
    output reg               busy,
    output reg               done,
    output reg  [WR-1:0]     root
);
    reg [2*WR-1:0] num;
    reg [WR+3:0]   rem;
    reg [WR-1:0]   r;
    reg [7:0]      cnt;

    wire [WR+3:0] rem_sh = {rem[WR+1:0], num[2*WR-1:2*WR-2]};
    wire [WR+3:0] trial  = {2'b00, r, 2'b01};

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            busy <= 1'b0;
            root <= {WR{1'b0}};
        end else if (start) begin
            num  <= n;
            rem  <= {(WR+4){1'b0}};
            r    <= {WR{1'b0}};
            cnt  <= WR;
            busy <= 1'b1;
        end else if (busy) begin
            if (cnt == 0) begin
                root <= r;
                busy <= 1'b0;
                done <= 1'b1;
            end else begin
                num <= num << 2;
                if (rem_sh >= trial) begin
                    rem <= rem_sh - trial;
                    r   <= {r[WR-2:0], 1'b1};
                end else begin
                    rem <= rem_sh;
                    r   <= {r[WR-2:0], 1'b0};
                end
                cnt <= cnt - 1'b1;
            end
        end
    end
endmodule

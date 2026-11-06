// Simple dual-port RAM: one synchronous write port, one synchronous read port.
// Read data appears on the clock edge after raddr is presented. Infers block RAM.
module sdp_ram #(
    parameter W  = 24,
    parameter AW = 11
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [W-1:0]  wdata,
    input  wire [AW-1:0] raddr,
    output reg  [W-1:0]  rdata
);
    reg [W-1:0] mem [0:(1<<AW)-1];

    always @(posedge clk) begin
        if (we)
            mem[waddr] <= wdata;
        rdata <= mem[raddr];
    end
endmodule

module feature_vector_generator(
    input wire clk,
    input wire reset,
    input wire [7:0] alpha,
    input wire [7:0] beta,
    input wire [7:0] theta,
    input wire [7:0] gsr,
    output reg [31:0] vector_out
);
always @(posedge clk or posedge reset) begin
    if (reset) begin
        vector_out <= 32'b0;
    end else begin
        vector_out <= {alpha, beta, theta, gsr};
    end
end
endmodule
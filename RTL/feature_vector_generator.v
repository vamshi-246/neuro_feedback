module feature_vector_generator #(
    parameter input_width = 8
)(
    input wire clk,
    input wire reset,
    input wire [input_width-1:0] alpha,
    input wire [input_width-1:0] beta,
    input wire [input_width-1:0] theta,
    input wire [input_width-1:0] gsr,
    output reg [input_width*4-1:0] vector_out
);
always @(posedge clk or posedge reset) begin
    if (reset) begin
        vector_out <= {(input_width * 4){1'b0}};
    end else begin
        vector_out <= {alpha, beta, theta, gsr};
    end
end
endmodule

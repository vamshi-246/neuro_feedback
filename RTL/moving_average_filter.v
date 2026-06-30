module moving_average_filter #(
    parameter input_width = 7
)(
    input  wire       clk,
    input  wire       reset,
    input  wire [input_width-1:0] rhythm,
    output reg  [input_width-1:0] smoothed_rhythm
);

    reg [input_width-1:0] delay_1;
    reg [input_width-1:0] delay_2;
    reg [input_width-1:0] delay_3;

    wire [input_width+1:0] sum;
    wire [input_width-1:0] average_result;

    assign sum = {2'b00, rhythm} +
                 {2'b00, delay_1} +
                 {2'b00, delay_2} +
                 {2'b00, delay_3};

    assign average_result = sum[8:2];

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            delay_1          <= 8'd0;
            delay_2          <= 8'd0;
            delay_3          <= 8'd0;
            smoothed_rhythm  <= 8'd0;
        end else begin
            delay_1          <= rhythm;
            delay_2          <= delay_1;
            delay_3          <= delay_2;
            smoothed_rhythm  <= average_result;
        end
    end
endmodule

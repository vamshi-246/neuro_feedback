module moving_average_filter #(
    parameter input_width = 8
)(
    input  wire       clk,
    input  wire       reset,
    input  wire [input_width-1:0] rhythm,
    output reg  [input_width-1:0] smoothed_rhythm
);

    localparam SUM_WIDTH = input_width + 2;

    reg [input_width-1:0] delay_1;
    reg [input_width-1:0] delay_2;
    reg [input_width-1:0] delay_3;

    wire [SUM_WIDTH-1:0] sum;
    wire [input_width-1:0] average_result;

    assign sum = {{2{1'b0}}, rhythm} +
                 {{2{1'b0}}, delay_1} +
                 {{2{1'b0}}, delay_2} +
                 {{2{1'b0}}, delay_3};

    assign average_result = sum >> 2;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            delay_1          <= {input_width{1'b0}};
            delay_2          <= {input_width{1'b0}};
            delay_3          <= {input_width{1'b0}};
            smoothed_rhythm  <= {input_width{1'b0}};
        end else begin
            delay_1          <= rhythm;
            delay_2          <= delay_1;
            delay_3          <= delay_2;
            smoothed_rhythm  <= average_result;
        end
    end
endmodule

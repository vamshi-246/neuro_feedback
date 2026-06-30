module power (
    input  wire       clk,
    input  wire       reset,
    input  wire [7:0] feature_signal,
    output reg  [15:0] power_out,
    output reg        power_valid
);

    parameter WINDOW_SIZE = 16;
    parameter SHIFT_BITS  = 4;

    reg [15:0] square_buffer [0:WINDOW_SIZE-1];
    reg [3:0]  write_index;
    reg [4:0]  sample_count;
    reg [19:0] sum_sq;

    wire [15:0] newest_square;
    wire [15:0] oldest_square;
    wire [19:0] updated_sum;

    integer i;

    assign newest_square = feature_signal * feature_signal;
    assign oldest_square = square_buffer[write_index];

    assign updated_sum = sum_sq + newest_square - oldest_square;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            write_index <= 0;
            sample_count <= 0;
            sum_sq <= 0;
            power_out <= 0;
            power_valid <= 0;

            for (i = 0; i < WINDOW_SIZE; i = i + 1) begin
                square_buffer[i] <= 0;
            end
        end else begin
            square_buffer[write_index] <= newest_square;
            sum_sq <= updated_sum;

            if (write_index == WINDOW_SIZE - 1)
                write_index <= 0;
            else
                write_index <= write_index + 1;

            if (sample_count < WINDOW_SIZE)
                sample_count <= sample_count + 1;
            
            if (sample_count == WINDOW_SIZE - 1) begin
                power_valid <= 1;
            end

            if (sample_count >= WINDOW_SIZE - 1) begin
                power_out <= updated_sum >> SHIFT_BITS;
            end
        end
    end

endmodule
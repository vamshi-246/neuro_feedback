module power #(
    parameter input_width = 8,
    parameter WINDOW_SIZE  = 16
)(
    input  wire       clk,
    input  wire       reset,
    input  wire [input_width-1:0] feature_signal,
    output reg  [input_width*2-1:0] power_out,
    output reg        power_valid
);

    function integer clog2;
        input integer value;
        integer temp;
        begin
            temp = value - 1;
            for (clog2 = 0; temp > 0; clog2 = clog2 + 1)
                temp = temp >> 1;
        end
    endfunction

    localparam SQUARE_WIDTH = input_width * 2;
    localparam INDEX_WIDTH   = (WINDOW_SIZE > 1) ? clog2(WINDOW_SIZE) : 1;
    localparam ACC_WIDTH     = SQUARE_WIDTH + clog2(WINDOW_SIZE);
    localparam SHIFT_BITS    = clog2(WINDOW_SIZE);

    reg [SQUARE_WIDTH-1:0] square_buffer [0:WINDOW_SIZE-1];
    reg [INDEX_WIDTH-1:0]  write_index;
    reg [INDEX_WIDTH:0]    sample_count;
    reg [ACC_WIDTH-1:0]    sum_sq;

    wire [SQUARE_WIDTH-1:0] newest_square;
    wire [SQUARE_WIDTH-1:0] oldest_square;
    wire [ACC_WIDTH-1:0]    updated_sum;

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

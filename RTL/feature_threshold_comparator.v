module feature_threshold_comparator #(
    parameter input_width = 8,
    parameter [input_width-1:0] low_max = 8'd35,
    parameter [input_width-1:0] high_min = 8'd65,
    parameter reverse_scale = 1'b0
)(
    input  wire [input_width-1:0] feature_value,
    output reg  [1:0]             pain_code
);

    always @(*) begin
        if (reverse_scale) begin
            if (feature_value >= high_min) begin
                pain_code = 2'd0;
            end else if (feature_value > low_max) begin
                pain_code = 2'd1;
            end else begin
                pain_code = 2'd2;
            end
        end else begin
            if (feature_value <= low_max) begin
                pain_code = 2'd0;
            end else if (feature_value < high_min) begin
                pain_code = 2'd1;
            end else begin
                pain_code = 2'd2;
            end
        end
    end

endmodule

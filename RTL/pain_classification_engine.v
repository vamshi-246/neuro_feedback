module pain_classification_engine #(
    parameter input_width = 8,
    parameter [input_width-1:0] alpha_low_max  = 8'd34,
    parameter [input_width-1:0] alpha_high_min = 8'd65,
    parameter [input_width-1:0] beta_low_max   = 8'd35,
    parameter [input_width-1:0] beta_high_min  = 8'd65,
    parameter [input_width-1:0] theta_low_max  = 8'd35,
    parameter [input_width-1:0] theta_high_min = 8'd55,
    parameter [input_width-1:0] gsr_low_max    = 8'd30,
    parameter [input_width-1:0] gsr_high_min   = 8'd65
)(
    input  wire                     clk,
    input  wire                     reset,
    input  wire [input_width*4-1:0] feature_vector,
    output wire [4:0]               pain_score,
    output wire [1:0]               raw_pain_level,
    output wire [1:0]               pain_state
);

    wire [input_width-1:0] alpha_feature;
    wire [input_width-1:0] beta_feature;
    wire [input_width-1:0] theta_feature;
    wire [input_width-1:0] gsr_feature;

    wire [1:0] alpha_code;
    wire [1:0] beta_code;
    wire [1:0] theta_code;
    wire [1:0] gsr_code;

    assign alpha_feature = feature_vector[(input_width*4)-1:(input_width*3)];
    assign beta_feature  = feature_vector[(input_width*3)-1:(input_width*2)];
    assign theta_feature = feature_vector[(input_width*2)-1:input_width];
    assign gsr_feature   = feature_vector[input_width-1:0];

    feature_threshold_comparator #(
        .input_width(input_width),
        .low_max(alpha_low_max),
        .high_min(alpha_high_min),
        .reverse_scale(1'b1)
    ) alpha_comparator (
        .feature_value(alpha_feature),
        .pain_code(alpha_code)
    );

    feature_threshold_comparator #(
        .input_width(input_width),
        .low_max(beta_low_max),
        .high_min(beta_high_min),
        .reverse_scale(1'b0)
    ) beta_comparator (
        .feature_value(beta_feature),
        .pain_code(beta_code)
    );

    feature_threshold_comparator #(
        .input_width(input_width),
        .low_max(theta_low_max),
        .high_min(theta_high_min),
        .reverse_scale(1'b0)
    ) theta_comparator (
        .feature_value(theta_feature),
        .pain_code(theta_code)
    );

    feature_threshold_comparator #(
        .input_width(input_width),
        .low_max(gsr_low_max),
        .high_min(gsr_high_min),
        .reverse_scale(1'b0)
    ) gsr_comparator (
        .feature_value(gsr_feature),
        .pain_code(gsr_code)
    );

    pain_classification_logic decision_logic (
        .alpha_code(alpha_code),
        .beta_code(beta_code),
        .theta_code(theta_code),
        .gsr_code(gsr_code),
        .pain_score(pain_score),
        .raw_pain_level(raw_pain_level)
    );

    pain_classifier_fsm pain_state_fsm (
        .clk(clk),
        .reset(reset),
        .raw_pain_level(raw_pain_level),
        .pain_score(pain_score),
        .pain_state(pain_state)
    );

endmodule

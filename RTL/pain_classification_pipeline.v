module pain_classification_pipeline #(
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
    input  wire [input_width-1:0]   alpha,
    input  wire [input_width-1:0]   beta,
    input  wire [input_width-1:0]   theta,
    input  wire [input_width-1:0]   gsr,
    output wire [input_width*4-1:0] feature_vector,
    output wire [4:0]               pain_score,
    output wire [1:0]               raw_pain_level,
    output wire [1:0]               pain_state
);

    feature_vector_generator #(
        .input_width(input_width)
    ) feature_vector_builder (
        .clk(clk),
        .reset(reset),
        .alpha(alpha),
        .beta(beta),
        .theta(theta),
        .gsr(gsr),
        .vector_out(feature_vector)
    );

    pain_classification_engine #(
        .input_width(input_width),
        .alpha_low_max(alpha_low_max),
        .alpha_high_min(alpha_high_min),
        .beta_low_max(beta_low_max),
        .beta_high_min(beta_high_min),
        .theta_low_max(theta_low_max),
        .theta_high_min(theta_high_min),
        .gsr_low_max(gsr_low_max),
        .gsr_high_min(gsr_high_min)
    ) pain_engine (
        .clk(clk),
        .reset(reset),
        .feature_vector(feature_vector),
        .pain_score(pain_score),
        .raw_pain_level(raw_pain_level),
        .pain_state(pain_state)
    );

endmodule

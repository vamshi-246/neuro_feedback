module pain_classification_system_top #(
    parameter STATE_DURATION  = 100,
    parameter POWER_WINDOW_SIZE = 16,
    parameter [7:0] ALPHA_LOW_MAX  = 8'd34,
    parameter [7:0] ALPHA_HIGH_MIN = 8'd65,
    parameter [7:0] BETA_LOW_MAX   = 8'd35,
    parameter [7:0] BETA_HIGH_MIN  = 8'd65,
    parameter [7:0] THETA_LOW_MAX  = 8'd35,
    parameter [7:0] THETA_HIGH_MIN = 8'd55,
    parameter [7:0] GSR_LOW_MAX    = 8'd30,
    parameter [7:0] GSR_HIGH_MIN   = 8'd65
)(
    input  wire       clk,
    input  wire       reset,
    output wire [1:0] synthetic_pain_state,
    output wire [7:0] alpha_raw,
    output wire [7:0] beta_raw,
    output wire [7:0] theta_raw,
    output wire [7:0] gsr_raw,
    output wire [7:0] alpha_filtered,
    output wire [7:0] beta_filtered,
    output wire [7:0] theta_filtered,
    output wire [7:0] gsr_filtered,
    output wire [15:0] alpha_window_power,
    output wire [15:0] beta_window_power,
    output wire [15:0] theta_window_power,
    output wire [15:0] gsr_window_power,
    output wire       alpha_power_valid,
    output wire       beta_power_valid,
    output wire       theta_power_valid,
    output wire       gsr_power_valid,
    output wire       preprocessing_valid,
    output wire [31:0] feature_vector,
    output wire [4:0] pain_score,
    output wire [1:0] raw_pain_level,
    output wire [1:0] classified_pain_state
);

    synthetic_signal_generator #(
        .STATE_DURATION(STATE_DURATION)
    ) signal_source (
        .clk(clk),
        .reset(reset),
        .alpha_power(alpha_raw),
        .beta_power(beta_raw),
        .theta_power(theta_raw),
        .gsr_level(gsr_raw),
        .pain_state(synthetic_pain_state)
    );

    moving_average_filter #(
        .input_width(8)
    ) alpha_filter (
        .clk(clk),
        .reset(reset),
        .rhythm(alpha_raw),
        .smoothed_rhythm(alpha_filtered)
    );

    moving_average_filter #(
        .input_width(8)
    ) beta_filter (
        .clk(clk),
        .reset(reset),
        .rhythm(beta_raw),
        .smoothed_rhythm(beta_filtered)
    );

    moving_average_filter #(
        .input_width(8)
    ) theta_filter (
        .clk(clk),
        .reset(reset),
        .rhythm(theta_raw),
        .smoothed_rhythm(theta_filtered)
    );

    moving_average_filter #(
        .input_width(8)
    ) gsr_filter (
        .clk(clk),
        .reset(reset),
        .rhythm(gsr_raw),
        .smoothed_rhythm(gsr_filtered)
    );

    // The synthetic generator already emits per-feature magnitudes, so these
    // power blocks are exposed for waveform inspection rather than used as
    // classifier inputs.
    power #(
        .input_width(8),
        .WINDOW_SIZE(POWER_WINDOW_SIZE)
    ) alpha_power_monitor (
        .clk(clk),
        .reset(reset),
        .feature_signal(alpha_filtered),
        .power_out(alpha_window_power),
        .power_valid(alpha_power_valid)
    );

    power #(
        .input_width(8),
        .WINDOW_SIZE(POWER_WINDOW_SIZE)
    ) beta_power_monitor (
        .clk(clk),
        .reset(reset),
        .feature_signal(beta_filtered),
        .power_out(beta_window_power),
        .power_valid(beta_power_valid)
    );

    power #(
        .input_width(8),
        .WINDOW_SIZE(POWER_WINDOW_SIZE)
    ) theta_power_monitor (
        .clk(clk),
        .reset(reset),
        .feature_signal(theta_filtered),
        .power_out(theta_window_power),
        .power_valid(theta_power_valid)
    );

    power #(
        .input_width(8),
        .WINDOW_SIZE(POWER_WINDOW_SIZE)
    ) gsr_power_monitor (
        .clk(clk),
        .reset(reset),
        .feature_signal(gsr_filtered),
        .power_out(gsr_window_power),
        .power_valid(gsr_power_valid)
    );

    assign preprocessing_valid = alpha_power_valid &
                                 beta_power_valid &
                                 theta_power_valid &
                                 gsr_power_valid;

    feature_vector_generator #(
        .input_width(8)
    ) feature_vector_builder (
        .clk(clk),
        .reset(reset),
        .alpha(alpha_filtered),
        .beta(beta_filtered),
        .theta(theta_filtered),
        .gsr(gsr_filtered),
        .vector_out(feature_vector)
    );

    pain_classification_engine #(
        .input_width(8),
        .alpha_low_max(ALPHA_LOW_MAX),
        .alpha_high_min(ALPHA_HIGH_MIN),
        .beta_low_max(BETA_LOW_MAX),
        .beta_high_min(BETA_HIGH_MIN),
        .theta_low_max(THETA_LOW_MAX),
        .theta_high_min(THETA_HIGH_MIN),
        .gsr_low_max(GSR_LOW_MAX),
        .gsr_high_min(GSR_HIGH_MIN)
    ) pain_engine (
        .clk(clk),
        .reset(reset),
        .feature_vector(feature_vector),
        .pain_score(pain_score),
        .raw_pain_level(raw_pain_level),
        .pain_state(classified_pain_state)
    );

endmodule

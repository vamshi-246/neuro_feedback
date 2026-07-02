`timescale 1ns/1ps

module pain_classification_system_top_tb;

    reg clk;
    reg reset;

    wire [1:0] synthetic_pain_state;
    wire [7:0] alpha_raw;
    wire [7:0] beta_raw;
    wire [7:0] theta_raw;
    wire [7:0] gsr_raw;
    wire [7:0] alpha_filtered;
    wire [7:0] beta_filtered;
    wire [7:0] theta_filtered;
    wire [7:0] gsr_filtered;
    wire [15:0] alpha_window_power;
    wire [15:0] beta_window_power;
    wire [15:0] theta_window_power;
    wire [15:0] gsr_window_power;
    wire alpha_power_valid;
    wire beta_power_valid;
    wire theta_power_valid;
    wire gsr_power_valid;
    wire preprocessing_valid;
    wire [31:0] feature_vector;
    wire [4:0] pain_score;
    wire [1:0] raw_pain_level;
    wire [1:0] classified_pain_state;

    integer error_count;

    pain_classification_system_top #(
        .STATE_DURATION(20),
        .POWER_WINDOW_SIZE(4)
    ) dut (
        .clk(clk),
        .reset(reset),
        .synthetic_pain_state(synthetic_pain_state),
        .alpha_raw(alpha_raw),
        .beta_raw(beta_raw),
        .theta_raw(theta_raw),
        .gsr_raw(gsr_raw),
        .alpha_filtered(alpha_filtered),
        .beta_filtered(beta_filtered),
        .theta_filtered(theta_filtered),
        .gsr_filtered(gsr_filtered),
        .alpha_window_power(alpha_window_power),
        .beta_window_power(beta_window_power),
        .theta_window_power(theta_window_power),
        .gsr_window_power(gsr_window_power),
        .alpha_power_valid(alpha_power_valid),
        .beta_power_valid(beta_power_valid),
        .theta_power_valid(theta_power_valid),
        .gsr_power_valid(gsr_power_valid),
        .preprocessing_valid(preprocessing_valid),
        .feature_vector(feature_vector),
        .pain_score(pain_score),
        .raw_pain_level(raw_pain_level),
        .classified_pain_state(classified_pain_state)
    );

    always #5 clk = ~clk;

    task automatic check_snapshot;
        input [1:0] expected_synthetic_state;
        input [1:0] expected_raw_level;
        input [1:0] expected_classified_state;
        input [4:0] expected_score;
        begin
            if (synthetic_pain_state !== expected_synthetic_state) begin
                $display("FAIL: expected synthetic state=%0d got=%0d",
                         expected_synthetic_state, synthetic_pain_state);
                error_count = error_count + 1;
            end

            if (raw_pain_level !== expected_raw_level) begin
                $display("FAIL: expected raw pain level=%0d got=%0d",
                         expected_raw_level, raw_pain_level);
                error_count = error_count + 1;
            end

            if (classified_pain_state !== expected_classified_state) begin
                $display("FAIL: expected classified pain state=%0d got=%0d",
                         expected_classified_state, classified_pain_state);
                error_count = error_count + 1;
            end

            if (pain_score !== expected_score) begin
                $display("FAIL: expected pain score=%0d got=%0d",
                         expected_score, pain_score);
                error_count = error_count + 1;
            end

            if (!preprocessing_valid) begin
                $display("FAIL: preprocessing_valid was not asserted");
                error_count = error_count + 1;
            end

            if (feature_vector !== {alpha_filtered, beta_filtered, theta_filtered, gsr_filtered}) begin
                $display("FAIL: feature vector does not match filtered outputs");
                error_count = error_count + 1;
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        reset = 1'b1;
        error_count = 0;

        #20;
        reset = 1'b0;

        repeat (12) @(posedge clk);
        #1;
        check_snapshot(2'd0, 2'd0, 2'd0, 5'd0);

        wait (synthetic_pain_state == 2'd1);
        repeat (8) @(posedge clk);
        #1;
        check_snapshot(2'd1, 2'd1, 2'd1, 5'd8);

        wait (synthetic_pain_state == 2'd2);
        repeat (8) @(posedge clk);
        #1;
        check_snapshot(2'd2, 2'd2, 2'd2, 5'd16);

        wait (synthetic_pain_state == 2'd0);
        repeat (8) @(posedge clk);
        #1;
        check_snapshot(2'd0, 2'd0, 2'd0, 5'd0);

        if (error_count == 0) begin
            $display("PASS: top-level pain classification system simulation checks succeeded.");
        end else begin
            $display("FAIL: top-level pain classification system simulation found %0d error(s).",
                     error_count);
        end

        $finish;
    end

    initial begin
        $monitor("Time=%0t | synth_state=%0d | raw=(%0d,%0d,%0d,%0d) | filt=(%0d,%0d,%0d,%0d) | score=%0d | raw_level=%0d | class_state=%0d | pre_valid=%0b",
                 $time,
                 synthetic_pain_state,
                 alpha_raw, beta_raw, theta_raw, gsr_raw,
                 alpha_filtered, beta_filtered, theta_filtered, gsr_filtered,
                 pain_score, raw_pain_level, classified_pain_state, preprocessing_valid);
    end

endmodule

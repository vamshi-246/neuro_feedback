`timescale 1ns/1ps

module pain_classification_pipeline_tb;

    parameter input_width = 8;

    reg clk;
    reg reset;
    reg [input_width-1:0] alpha;
    reg [input_width-1:0] beta;
    reg [input_width-1:0] theta;
    reg [input_width-1:0] gsr;

    wire [input_width*4-1:0] feature_vector;
    wire [4:0] pain_score;
    wire [1:0] raw_pain_level;
    wire [1:0] pain_state;

    integer error_count;

    pain_classification_pipeline #(
        .input_width(input_width),
        .alpha_low_max(8'd34),
        .alpha_high_min(8'd65),
        .beta_low_max(8'd35),
        .beta_high_min(8'd65),
        .theta_low_max(8'd35),
        .theta_high_min(8'd55),
        .gsr_low_max(8'd30),
        .gsr_high_min(8'd65)
    ) dut (
        .clk(clk),
        .reset(reset),
        .alpha(alpha),
        .beta(beta),
        .theta(theta),
        .gsr(gsr),
        .feature_vector(feature_vector),
        .pain_score(pain_score),
        .raw_pain_level(raw_pain_level),
        .pain_state(pain_state)
    );

    always #5 clk = ~clk;

    task automatic drive_vector;
        input [input_width-1:0] alpha_value;
        input [input_width-1:0] beta_value;
        input [input_width-1:0] theta_value;
        input [input_width-1:0] gsr_value;
        begin
            @(negedge clk);
            alpha = alpha_value;
            beta  = beta_value;
            theta = theta_value;
            gsr   = gsr_value;
        end
    endtask

    task automatic check_raw_outputs;
        input [input_width-1:0] alpha_value;
        input [input_width-1:0] beta_value;
        input [input_width-1:0] theta_value;
        input [input_width-1:0] gsr_value;
        input [1:0] expected_alpha_code;
        input [1:0] expected_beta_code;
        input [1:0] expected_theta_code;
        input [1:0] expected_gsr_code;
        input [4:0] expected_score;
        input [1:0] expected_raw_level;
        begin
            @(posedge clk);
            #1;

            if (feature_vector !== {alpha_value, beta_value, theta_value, gsr_value}) begin
                $display("FAIL: expected feature vector=%h got=%h",
                         {alpha_value, beta_value, theta_value, gsr_value}, feature_vector);
                error_count = error_count + 1;
            end

            if (dut.pain_engine.alpha_code !== expected_alpha_code) begin
                $display("FAIL: expected alpha_code=%0d got=%0d",
                         expected_alpha_code, dut.pain_engine.alpha_code);
                error_count = error_count + 1;
            end

            if (dut.pain_engine.beta_code !== expected_beta_code) begin
                $display("FAIL: expected beta_code=%0d got=%0d",
                         expected_beta_code, dut.pain_engine.beta_code);
                error_count = error_count + 1;
            end

            if (dut.pain_engine.theta_code !== expected_theta_code) begin
                $display("FAIL: expected theta_code=%0d got=%0d",
                         expected_theta_code, dut.pain_engine.theta_code);
                error_count = error_count + 1;
            end

            if (dut.pain_engine.gsr_code !== expected_gsr_code) begin
                $display("FAIL: expected gsr_code=%0d got=%0d",
                         expected_gsr_code, dut.pain_engine.gsr_code);
                error_count = error_count + 1;
            end

            if (pain_score !== expected_score) begin
                $display("FAIL: expected pain_score=%0d got=%0d",
                         expected_score, pain_score);
                error_count = error_count + 1;
            end

            if (raw_pain_level !== expected_raw_level) begin
                $display("FAIL: expected raw_pain_level=%0d got=%0d",
                         expected_raw_level, raw_pain_level);
                error_count = error_count + 1;
            end
        end
    endtask

    task automatic check_state_after_second_clock;
        input [1:0] expected_state;
        begin
            @(posedge clk);
            #1;

            if (pain_state !== expected_state) begin
                $display("FAIL: expected pain_state=%0d got=%0d",
                         expected_state, pain_state);
                error_count = error_count + 1;
            end
        end
    endtask

    task automatic apply_vector_and_check;
        input [input_width-1:0] alpha_value;
        input [input_width-1:0] beta_value;
        input [input_width-1:0] theta_value;
        input [input_width-1:0] gsr_value;
        input [1:0] expected_alpha_code;
        input [1:0] expected_beta_code;
        input [1:0] expected_theta_code;
        input [1:0] expected_gsr_code;
        input [4:0] expected_score;
        input [1:0] expected_raw_level;
        input [1:0] expected_state;
        begin
            drive_vector(alpha_value, beta_value, theta_value, gsr_value);
            check_raw_outputs(
                alpha_value,
                beta_value,
                theta_value,
                gsr_value,
                expected_alpha_code,
                expected_beta_code,
                expected_theta_code,
                expected_gsr_code,
                expected_score,
                expected_raw_level
            );
            check_state_after_second_clock(expected_state);
        end
    endtask

    initial begin
        clk = 1'b0;
        reset = 1'b1;
        alpha = {input_width{1'b0}};
        beta  = {input_width{1'b0}};
        theta = {input_width{1'b0}};
        gsr   = {input_width{1'b0}};
        error_count = 0;

        @(posedge clk);
        @(posedge clk);
        @(negedge clk);
        alpha = 8'd80;
        beta  = 8'd20;
        theta = 8'd30;
        gsr   = 8'd15;
        reset = 1'b0;

        check_raw_outputs(8'd80, 8'd20, 8'd30, 8'd15, 2'd0, 2'd0, 2'd0, 2'd0, 5'd0, 2'd0);
        check_state_after_second_clock(2'd0);

        apply_vector_and_check(8'd50, 8'd50, 8'd40, 8'd45, 2'd1, 2'd1, 2'd1, 2'd1, 5'd8, 2'd1, 2'd1);
        apply_vector_and_check(8'd20, 8'd80, 8'd60, 8'd85, 2'd2, 2'd2, 2'd2, 2'd2, 5'd16, 2'd2, 2'd2);

        drive_vector(8'd80, 8'd20, 8'd30, 8'd15);
        check_raw_outputs(8'd80, 8'd20, 8'd30, 8'd15, 2'd0, 2'd0, 2'd0, 2'd0, 5'd0, 2'd0);
        check_state_after_second_clock(2'd1);

        drive_vector(8'd80, 8'd20, 8'd30, 8'd15);
        check_raw_outputs(8'd80, 8'd20, 8'd30, 8'd15, 2'd0, 2'd0, 2'd0, 2'd0, 5'd0, 2'd0);
        check_state_after_second_clock(2'd0);

        drive_vector(8'd20, 8'd80, 8'd40, 8'd45);
        check_raw_outputs(8'd20, 8'd80, 8'd40, 8'd45, 2'd2, 2'd2, 2'd1, 2'd1, 5'd12, 2'd2);
        check_state_after_second_clock(2'd1);

        drive_vector(8'd20, 8'd80, 8'd40, 8'd45);
        check_raw_outputs(8'd20, 8'd80, 8'd40, 8'd45, 2'd2, 2'd2, 2'd1, 2'd1, 5'd12, 2'd2);
        check_state_after_second_clock(2'd2);

        drive_vector(8'd50, 8'd50, 8'd40, 8'd45);
        check_raw_outputs(8'd50, 8'd50, 8'd40, 8'd45, 2'd1, 2'd1, 2'd1, 2'd1, 5'd8, 2'd1);
        check_state_after_second_clock(2'd2);

        drive_vector(8'd50, 8'd50, 8'd40, 8'd45);
        check_raw_outputs(8'd50, 8'd50, 8'd40, 8'd45, 2'd1, 2'd1, 2'd1, 2'd1, 5'd8, 2'd1);
        check_state_after_second_clock(2'd1);

        apply_vector_and_check(8'd20, 8'd80, 8'd60, 8'd85, 2'd2, 2'd2, 2'd2, 2'd2, 5'd16, 2'd2, 2'd2);

        apply_vector_and_check(8'd80, 8'd20, 8'd55, 8'd10, 2'd0, 2'd0, 2'd2, 2'd0, 5'd2, 2'd0, 2'd1);

        drive_vector(8'd80, 8'd20, 8'd30, 8'd15);
        check_raw_outputs(8'd80, 8'd20, 8'd30, 8'd15, 2'd0, 2'd0, 2'd0, 2'd0, 5'd0, 2'd0);
        check_state_after_second_clock(2'd0);

        if (error_count == 0) begin
            $display("PASS: pain classification pipeline checks succeeded.");
        end else begin
            $display("FAIL: pain classification pipeline found %0d error(s).", error_count);
        end

        $finish;
    end

    initial begin
        $monitor("Time=%0t | alpha=%0d | beta=%0d | theta=%0d | gsr=%0d | vector=%h | score=%0d | raw=%0d | state=%0d",
                 $time, alpha, beta, theta, gsr, feature_vector, pain_score, raw_pain_level, pain_state);
    end

endmodule

`timescale 1ns/1ps

module power_tb;

    parameter input_width = 8;
    parameter WINDOW_SIZE = 4;

    reg clk;
    reg reset;
    reg [input_width-1:0] feature_signal;

    wire [input_width*2-1:0] power_out;
    wire power_valid;

    integer error_count;

    power #(
        .input_width(input_width),
        .WINDOW_SIZE(WINDOW_SIZE)
    ) dut (
        .clk(clk),
        .reset(reset),
        .feature_signal(feature_signal),
        .power_out(power_out),
        .power_valid(power_valid)
    );

    always #5 clk = ~clk;

    task automatic apply_sample_and_check;
        input [input_width-1:0] sample_value;
        input                   expected_valid;
        input [input_width*2-1:0] expected_power;
        begin
            @(negedge clk);
            feature_signal = sample_value;
            @(posedge clk);
            #1;

            if (power_valid !== expected_valid) begin
                $display("FAIL: sample=%0d expected valid=%0b got=%0b",
                         sample_value, expected_valid, power_valid);
                error_count = error_count + 1;
            end

            if (expected_valid && (power_out !== expected_power)) begin
                $display("FAIL: sample=%0d expected power=%0d got=%0d",
                         sample_value, expected_power, power_out);
                error_count = error_count + 1;
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        reset = 1'b1;
        feature_signal = {input_width{1'b0}};
        error_count = 0;

        @(posedge clk);
        @(posedge clk);
        @(negedge clk);
        feature_signal = 8'd2;
        reset = 1'b0;

        apply_sample_and_check(8'd4, 1'b0, 16'd0);
        apply_sample_and_check(8'd6, 1'b0, 16'd0);
        apply_sample_and_check(8'd8, 1'b1, 16'd30);
        apply_sample_and_check(8'd10, 1'b1, 16'd54);
        apply_sample_and_check(8'd4, 1'b1, 16'd54);
        apply_sample_and_check(8'd0, 1'b1, 16'd45);

        if (error_count == 0) begin
            $display("PASS: power.v sliding-window power checks succeeded.");
        end else begin
            $display("FAIL: power.v testbench found %0d error(s).", error_count);
        end

        $finish;
    end

    initial begin
        $monitor("Time=%0t | signal=%0d | valid=%0b | power=%0d",
                 $time, feature_signal, power_valid, power_out);
    end

endmodule

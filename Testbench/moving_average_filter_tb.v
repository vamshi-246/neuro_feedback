`timescale 1ns/1ps

module moving_average_filter_tb;

    parameter input_width = 8;

    reg clk;
    reg reset;
    reg [input_width-1:0] rhythm;

    wire [input_width-1:0] smoothed_rhythm;

    moving_average_filter #(
        .input_width(input_width)
    ) dut (
        .clk(clk),
        .reset(reset),
        .rhythm(rhythm),
        .smoothed_rhythm(smoothed_rhythm)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        reset = 1;
        rhythm = 0;

        #12;
        reset = 0;

        rhythm = 80; #10;
        rhythm = 78; #10;
        rhythm = 85; #10;
        rhythm = 81; #10;

        if (smoothed_rhythm != 81) begin
            $display("FAIL: expected 81, got %0d", smoothed_rhythm);
        end else begin
            $display("PASS: moving average output = %0d", smoothed_rhythm);
        end

        #20;
        $finish;
    end

    initial begin
        $monitor("Time=%0t | rhythm=%0d | smoothed_rhythm=%0d",
                 $time, rhythm, smoothed_rhythm);
    end

endmodule

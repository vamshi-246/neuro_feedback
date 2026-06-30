`timescale 1ns/1ps

module moving_average_filter_tb;

    reg clk;
    reg reset;
    reg [7:0] rhythm;

    wire [7:0] smoothed_rhythm;

    moving_average_filter dut (
        .clk(clk),
        .reset(reset),
        .rhythm(rhythm),
        .smoothed_rhythm(smoothed_rhythm)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        reset = 1;
        rhythm = 8'd0;

        #12;
        reset = 0;

        rhythm = 8'd80; #10;
        rhythm = 8'd78; #10;
        rhythm = 8'd85; #10;
        rhythm = 8'd81; #10;

        if (smoothed_rhythm != 8'd81) begin
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

`timescale 1ns/1ps

module synthetic_signal_generator_tb;

    reg clk;
    reg reset;

    wire [7:0] alpha_power;
    wire [7:0] beta_power;
    wire [7:0] theta_power;
    wire [7:0] gsr_level;
    wire [1:0] pain_state;

    synthetic_signal_generator dut (
        .clk(clk),
        .reset(reset),
        .alpha_power(alpha_power),
        .beta_power(beta_power),
        .theta_power(theta_power),
        .gsr_level(gsr_level),
        .pain_state(pain_state)
    );

    // Clock generation: 10 ns clock period
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        reset = 1;

        #20;
        reset = 0;

        // Run long enough to observe LOW, MEDIUM, HIGH, and repeat
        #3500;

        $finish;
    end

    initial begin
        $monitor("Time=%0t | State=%b | Alpha=%d | Beta=%d | Theta=%d | GSR=%d",
                  $time, pain_state, alpha_power, beta_power, theta_power, gsr_level);
    end

endmodule

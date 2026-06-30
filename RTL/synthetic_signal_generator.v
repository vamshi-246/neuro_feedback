module synthetic_signal_generator (
    input  wire       clk,
    input  wire       reset,

    output reg [7:0]  alpha_power,
    output reg [7:0]  beta_power,
    output reg [7:0]  theta_power,
    output reg [7:0]  gsr_level,
    output reg [1:0]  pain_state
);

    // State encoding
    localparam LOW_PAIN    = 2'b00;
    localparam MED_PAIN    = 2'b01;
    localparam HIGH_PAIN   = 2'b10;

    // Number of clock cycles to stay in each state
    parameter STATE_DURATION = 100;

    reg [7:0] counter;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            counter    <= 0;
            pain_state <= LOW_PAIN;
        end else begin
            if (counter == STATE_DURATION - 1) begin
                counter <= 0;

                case (pain_state)
                    LOW_PAIN:  pain_state <= MED_PAIN;
                    MED_PAIN:  pain_state <= HIGH_PAIN;
                    HIGH_PAIN: pain_state <= LOW_PAIN;
                    default:   pain_state <= LOW_PAIN;
                endcase

            end else begin
                counter <= counter + 1;
            end
        end
    end

    always @(*) begin
        case (pain_state)

            LOW_PAIN: begin
                alpha_power = 8'd80;
                beta_power  = 8'd20;
                theta_power = 8'd30;
                gsr_level   = 8'd15;
            end

            MED_PAIN: begin
                alpha_power = 8'd50;
                beta_power  = 8'd50;
                theta_power = 8'd40;
                gsr_level   = 8'd45;
            end

            HIGH_PAIN: begin
                alpha_power = 8'd20;
                beta_power  = 8'd80;
                theta_power = 8'd60;
                gsr_level   = 8'd85;
            end

            default: begin
                alpha_power = 8'd0;
                beta_power  = 8'd0;
                theta_power = 8'd0;
                gsr_level   = 8'd0;
            end

        endcase
    end
endmodule
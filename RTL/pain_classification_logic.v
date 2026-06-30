module pain_classification_logic(
    input  wire [1:0] alpha_code,
    input  wire [1:0] beta_code,
    input  wire [1:0] theta_code,
    input  wire [1:0] gsr_code,
    output reg  [4:0] pain_score,
    output reg  [1:0] raw_pain_level
);

    localparam LOW_PAIN       = 2'b00;
    localparam MODERATE_PAIN  = 2'b01;
    localparam HIGH_PAIN      = 2'b10;
    localparam SCORE_LOW_MAX  = 5'd4;
    localparam SCORE_HIGH_MIN = 5'd11;

    always @(*) begin
        pain_score = ({3'b000, alpha_code} << 1) +
                     ({3'b000, beta_code} << 1) +
                     {3'b000, theta_code} +
                     ({3'b000, gsr_code} * 3);

        if (pain_score <= SCORE_LOW_MAX) begin
            raw_pain_level = LOW_PAIN;
        end else if (pain_score >= SCORE_HIGH_MIN) begin
            raw_pain_level = HIGH_PAIN;
        end else begin
            raw_pain_level = MODERATE_PAIN;
        end
    end

endmodule

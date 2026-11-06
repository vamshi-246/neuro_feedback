module pain_classification_logic(
    input  wire [1:0] alpha_code,
    input  wire [1:0] beta_code,
    input  wire [1:0] theta_code,
    output reg  [4:0] pain_score,
    output reg  [1:0] raw_pain_level
);

    // GSR was dropped from this project's feature set (none of the candidate
    // EEG pain datasets include a skin-conductance channel -- see project
    // notes). Score range is now 0-10 (alpha/beta weight 2, theta weight 1,
    // each code 0-2): max = 2*2 + 2*2 + 1*2 = 10. Thresholds below preserve
    // the original formula's proportions (old: low<=4/16, high>=11/16 of a
    // 0-16 scale) on this smaller 0-10 scale.
    localparam LOW_PAIN       = 2'b00;
    localparam MODERATE_PAIN  = 2'b01;
    localparam HIGH_PAIN      = 2'b10;
    localparam SCORE_LOW_MAX  = 5'd2;
    localparam SCORE_HIGH_MIN = 5'd7;

    always @(*) begin
        pain_score = ({3'b000, alpha_code} << 1) +
                     ({3'b000, beta_code} << 1) +
                     {3'b000, theta_code};

        if (pain_score <= SCORE_LOW_MAX) begin
            raw_pain_level = LOW_PAIN;
        end else if (pain_score >= SCORE_HIGH_MIN) begin
            raw_pain_level = HIGH_PAIN;
        end else begin
            raw_pain_level = MODERATE_PAIN;
        end
    end

endmodule

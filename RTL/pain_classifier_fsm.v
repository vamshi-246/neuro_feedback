module pain_classifier_fsm(
    input  wire       clk,
    input  wire       reset,
    input  wire [1:0] raw_pain_level,
    input  wire [4:0] pain_score,
    output reg  [1:0] pain_state
);

    localparam LOW_PAIN          = 2'b00;
    localparam MODERATE_PAIN     = 2'b01;
    localparam HIGH_PAIN         = 2'b10;
    localparam SCORE_STRONG_HIGH = 5'd14;

    reg low_confirm_pending;
    reg moderate_confirm_pending;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            pain_state <= LOW_PAIN;
            low_confirm_pending <= 1'b0;
            moderate_confirm_pending <= 1'b0;
        end else begin
            case (pain_state)
                LOW_PAIN: begin
                    low_confirm_pending <= 1'b0;
                    moderate_confirm_pending <= 1'b0;

                    case (raw_pain_level)
                        HIGH_PAIN: begin
                            if (pain_score >= SCORE_STRONG_HIGH) begin
                                pain_state <= HIGH_PAIN;
                            end else begin
                                pain_state <= MODERATE_PAIN;
                            end
                        end
                        MODERATE_PAIN: pain_state <= MODERATE_PAIN;
                        default:       pain_state <= LOW_PAIN;
                    endcase
                end

                MODERATE_PAIN: begin
                    moderate_confirm_pending <= 1'b0;

                    case (raw_pain_level)
                        LOW_PAIN: begin
                            if (low_confirm_pending) begin
                                pain_state <= LOW_PAIN;
                                low_confirm_pending <= 1'b0;
                            end else begin
                                pain_state <= MODERATE_PAIN;
                                low_confirm_pending <= 1'b1;
                            end
                        end
                        HIGH_PAIN: begin
                            pain_state <= HIGH_PAIN;
                            low_confirm_pending <= 1'b0;
                        end
                        default: begin
                            pain_state <= MODERATE_PAIN;
                            low_confirm_pending <= 1'b0;
                        end
                    endcase
                end

                HIGH_PAIN: begin
                    low_confirm_pending <= 1'b0;

                    case (raw_pain_level)
                        LOW_PAIN: begin
                            pain_state <= MODERATE_PAIN;
                            low_confirm_pending <= 1'b1;
                            moderate_confirm_pending <= 1'b0;
                        end
                        MODERATE_PAIN: begin
                            if (moderate_confirm_pending) begin
                                pain_state <= MODERATE_PAIN;
                                moderate_confirm_pending <= 1'b0;
                            end else begin
                                pain_state <= HIGH_PAIN;
                                moderate_confirm_pending <= 1'b1;
                            end
                        end
                        default: begin
                            pain_state <= HIGH_PAIN;
                            moderate_confirm_pending <= 1'b0;
                        end
                    endcase
                end

                default: begin
                    pain_state <= LOW_PAIN;
                    low_confirm_pending <= 1'b0;
                    moderate_confirm_pending <= 1'b0;
                end
            endcase
        end
    end

endmodule

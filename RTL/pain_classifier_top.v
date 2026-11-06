// Pain classifier, no-MVAR model (three_class__no_mvar in outputs/deploy/deployment_model.json).
//
//   3 EEG channels (Cz, C3, C4) at 1 kHz + stimulus trigger
//     -> 4096-deep circular raw buffer (keeps running during processing)
//     -> 9 zero-phase filter runs (filtfilt_engine), each followed by a window scan
//        (feature_engine)
//     -> 16 (A, B) feature pairs -> 16 8-bit codes (threshold_quantizer)
//     -> 3 logits, argmax (linear_classifier)
//
// Epoch: 3000 samples, onset at sample PRE (default 1000, i.e. 1 s before and 2 s
// after the stimulus). The trigger is asserted together with s_valid on the onset
// sample. The class is ready 2 s after the trigger plus the processing time
// (~2.31 M clock cycles, about 46 ms at 50 MHz). Triggers that arrive before PRE
// samples of history exist, or while an epoch is in progress, are counted and dropped.
//
// Input sample format: signed 20 bit, LSB = 2^-10 uV (+-512 uV full scale).
module pain_classifier_top #(
    parameter PRE       = 1000,
    // ROM images from scripts/hardware/hw_golden.py --write-mem (RTL/mem/)
    parameter COEF_FILE = "coef.mem",
    parameter TH_FILE   = "thresh.mem",
    parameter W_FILE    = "weights.mem",
    parameter B_FILE    = "bias.mem"
) (
    input  wire               clk,
    input  wire               rst,
    // sample stream
    input  wire               s_valid,
    input  wire signed [19:0] s_cz,
    input  wire signed [19:0] s_c3,
    input  wire signed [19:0] s_c4,
    input  wire               s_trigger,
    // result
    output reg                result_valid,     // one-cycle pulse
    output reg  [1:0]         result_class,     // 0 Low, 1 Moderate, 2 High
    output reg  signed [39:0] logit_low,
    output reg  signed [39:0] logit_mod,
    output reg  signed [39:0] logit_high,
    output reg  [127:0]       feature_codes,    // q[i] = feature_codes[8*i +: 8]
    output wire               busy,
    output reg  [15:0]        triggers_dropped
);
    localparam N = 3000;

    // ------------------------------------------------------------ raw circular buffer
    reg  [11:0] wr_ptr;
    reg  [11:0] filled;              // saturates at PRE
    reg  [11:0] base;                // address of epoch sample 0
    reg  [11:0] raw_raddr;
    wire [59:0] raw_word;
    wire [11:0] ff_raw_idx;

    sdp_ram #(.W(60), .AW(12)) u_raw (
        .clk(clk), .we(s_valid), .waddr(wr_ptr), .wdata({s_c4, s_c3, s_cz}),
        .raddr(raw_raddr), .rdata(raw_word));

    always @(posedge clk) begin
        raw_raddr <= base + ff_raw_idx;                  // registered: 2-cycle read
        if (rst) begin
            wr_ptr <= 12'd0;
            filled <= 12'd0;
        end else if (s_valid) begin
            wr_ptr <= wr_ptr + 12'd1;
            if (filled < PRE) filled <= filled + 12'd1;
        end
    end

    // ------------------------------------------------------------ run table
    // run:      0    1      2      3     4      5      6     7      8
    //           ERP  delta  theta  beta  gamma  broad  beta  delta  delta
    // channel:  Cz   Cz     Cz     Cz    Cz     Cz     C4    C3     C4
    // buffer:   A    A      A      A     A      A      A     A      B
    reg [3:0] run;
    reg [2:0] run_band;
    reg [1:0] run_ch;
    always @(*) begin
        case (run)
        4'd0: begin run_band = 3'd4; run_ch = 2'd0; end
        4'd1: begin run_band = 3'd0; run_ch = 2'd0; end
        4'd2: begin run_band = 3'd1; run_ch = 2'd0; end
        4'd3: begin run_band = 3'd2; run_ch = 2'd0; end
        4'd4: begin run_band = 3'd3; run_ch = 2'd0; end
        4'd5: begin run_band = 3'd5; run_ch = 2'd0; end
        4'd6: begin run_band = 3'd2; run_ch = 2'd2; end
        4'd7: begin run_band = 3'd0; run_ch = 2'd1; end
        default: begin run_band = 3'd0; run_ch = 2'd2; end
        endcase
    end
    wire run_buf_b = (run == 4'd8);

    // ------------------------------------------------------------ filter engine
    reg                ff_start;
    wire               ff_we, ff_done;
    wire [10:0]        ff_addr;
    wire signed [23:0] ff_data;
    filtfilt_engine #(.PRE(PRE), .COEF_FILE(COEF_FILE)) u_ff (
        .clk(clk), .rst(rst), .start(ff_start), .band(run_band), .ch(run_ch),
        .raw_idx(ff_raw_idx), .raw_word(raw_word),
        .fb_we(ff_we), .fb_addr(ff_addr), .fb_data(ff_data), .done(ff_done));

    // ------------------------------------------------------------ feature buffers
    wire [10:0] fa_raddr, fb_raddr;
    wire [23:0] fa_rdata, fb_rdata;
    sdp_ram #(.W(24), .AW(11)) u_fba (
        .clk(clk), .we(ff_we & ~run_buf_b), .waddr(ff_addr), .wdata(ff_data),
        .raddr(fa_raddr), .rdata(fa_rdata));
    sdp_ram #(.W(24), .AW(11)) u_fbb (
        .clk(clk), .we(ff_we & run_buf_b), .waddr(ff_addr), .wdata(ff_data),
        .raddr(fb_raddr), .rdata(fb_rdata));

    // ------------------------------------------------------------ feature engine
    reg                fe_scan, fe_fin;
    wire               fe_done;
    wire [3:0]         q_sel;
    wire signed [71:0] fe_a, fe_b;
    feature_engine u_fe (
        .clk(clk), .rst(rst), .scan_start(fe_scan), .run(run), .fin_start(fe_fin),
        .ra_addr(fa_raddr), .ra_data(fa_rdata), .rb_addr(fb_raddr), .rb_data(fb_rdata),
        .done(fe_done), .sel(q_sel), .a_out(fe_a), .b_out(fe_b));

    // ------------------------------------------------------------ quantizer, classifier
    reg          q_start;
    wire         q_done;
    wire [127:0] q_flat;
    threshold_quantizer #(.TH_FILE(TH_FILE)) u_q (
        .clk(clk), .rst(rst), .start(q_start), .sel(q_sel), .a_in(fe_a), .b_in(fe_b),
        .q_flat(q_flat), .done(q_done));

    reg                lc_start;
    wire               lc_done;
    wire signed [39:0] l0, l1, l2;
    wire [1:0]         lc_cls;
    linear_classifier #(.W_FILE(W_FILE), .B_FILE(B_FILE)) u_lc (
        .clk(clk), .rst(rst), .start(lc_start), .q_flat(q_flat),
        .logit0(l0), .logit1(l1), .logit2(l2), .cls(lc_cls), .done(lc_done));

    // ------------------------------------------------------------ sequencer
    localparam ST_IDLE = 3'd0, ST_ARMED = 3'd1, ST_FF = 3'd2, ST_FE = 3'd3,
               ST_FIN = 3'd4, ST_Q = 3'd5, ST_LC = 3'd6;
    reg [2:0]  st;
    reg [11:0] post;                 // samples written since (and including) onset

    assign busy = (st != ST_IDLE);

    always @(posedge clk) begin
        ff_start     <= 1'b0;
        fe_scan      <= 1'b0;
        fe_fin       <= 1'b0;
        q_start      <= 1'b0;
        lc_start     <= 1'b0;
        result_valid <= 1'b0;
        if (rst) begin
            st               <= ST_IDLE;
            run              <= 4'd0;
            base             <= 12'd0;
            triggers_dropped <= 16'd0;
        end else begin
            if (s_valid && s_trigger && !(st == ST_IDLE && filled >= PRE))
                triggers_dropped <= triggers_dropped + 16'd1;

            case (st)
            ST_IDLE: if (s_valid && s_trigger && filled >= PRE) begin
                base <= wr_ptr - PRE;
                post <= 12'd1;
                st   <= ST_ARMED;
            end
            ST_ARMED: if (s_valid) begin
                if (post == N - PRE - 1) begin          // last epoch sample written now
                    run      <= 4'd0;
                    ff_start <= 1'b1;
                    st       <= ST_FF;
                end
                post <= post + 12'd1;
            end
            ST_FF: if (ff_done) begin
                fe_scan <= 1'b1;
                st      <= ST_FE;
            end
            ST_FE: if (fe_done) begin
                if (run == 4'd8) begin
                    fe_fin <= 1'b1;
                    st     <= ST_FIN;
                end else begin
                    run      <= run + 4'd1;
                    ff_start <= 1'b1;
                    st       <= ST_FF;
                end
            end
            ST_FIN: if (fe_done) begin
                q_start <= 1'b1;
                st      <= ST_Q;
            end
            ST_Q: if (q_done) begin
                lc_start <= 1'b1;
                st       <= ST_LC;
            end
            ST_LC: if (lc_done) begin
                result_valid  <= 1'b1;
                result_class  <= lc_cls;
                logit_low     <= l0;
                logit_mod     <= l1;
                logit_high    <= l2;
                feature_codes <= q_flat;
                st            <= ST_IDLE;
            end
            default: st <= ST_IDLE;
            endcase
        end
    end
endmodule

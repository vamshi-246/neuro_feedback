// Feature engine: turns the filtered feature buffers into the 16 (A, B) pairs.
//
// Each feature is later quantized by comparing A against c_k * B (see
// threshold_quantizer.v), so this block never divides, takes a log or a square
// root of a feature -- it only accumulates sums over windows and does a few exact
// wide multiplies at the end.  The one square root (coupling denominator) is integer
// and exact (floor).
//
// Feature-buffer index j = sample - (onset - 500), j = 0..1499. Windows in j:
//   baseline 0..499   w0 500..799   w1 800..1099   N2 search 650..799
//   N2 mean 650..749  P2 mean 750..949  P2 search n2+1..1049  RMS 500..1099
//   coupling / Hjorth 500..1499
//
// Runs (scan_start with `run`), matching hw_golden.RUNS:
//   0 ERP Cz   1 delta Cz   2 theta Cz   3 beta Cz   4 gamma Cz   5 broadband Cz
//   6 beta C4  7 delta C3   8 delta C4 (in buffer B; also computes C3-C4 coupling)
//
// (A, B) per feature, JSON column order (K300 = 300*2^20, K600 = 600*2^20):
//    0 Cz delta w0     P0[1], K300        8 Cz delta/gamma  P0[1], P0[4]
//    1 ERP RMS         Srms,  K600        9 Cz delta w1     P1,    K300
//    2 Cz theta w0     P0[2], K300       10 C4 delta w0     P0[8], K300
//    3 C3-C4 coupling  u,     isqrt(v*w) 11 C4 beta/delta   P0[6], P0[8]
//    4 P2 peak         p2,    2^10       12 P2 window mean  Sp2m,  200*2^10
//    5 N2-P2           p2-n2, 2^10       13 N2 window mean  Sn2m,  100*2^10
//    6 N2 peak         n2,    2^10       14 Cz beta/delta   P0[3], P0[1]
//    7 C3 delta w0     P0[7], K300       15 Hjorth mobility 999*Sdd-Sd^2, 1000*Sxx-Sx^2
module feature_engine (
    input  wire               clk,
    input  wire               rst,
    input  wire               scan_start,
    input  wire [3:0]         run,
    input  wire               fin_start,
    output reg  [10:0]        ra_addr,      // feature buffer A read (1-cycle RAM)
    input  wire signed [23:0] ra_data,
    output reg  [10:0]        rb_addr,      // feature buffer B read
    input  wire signed [23:0] rb_data,
    output reg                done,
    input  wire [3:0]         sel,
    output reg  signed [71:0] a_out,
    output reg  signed [71:0] b_out
);
    localparam [31:0] RECIP_500 = 32'd8589935;   // round(2^32 / 500)

    // ------------------------------------------------------------ square multiplier
    reg  signed [26:0] sm_a, sm_b;
    reg  signed [53:0] sq;
    always @(posedge clk) sq <= sm_a * sm_b;
    wire signed [63:0] sq64 = {{10{sq[53]}}, sq};

    // ------------------------------------------------------------ wide multiplier / sqrt
    reg                 mul_start;
    reg  signed [71:0]  mul_a, mul_b;
    wire                mul_done;
    wire signed [143:0] mul_p;
    mul_seq #(.WA(72), .WB(72)) u_mul (
        .clk(clk), .rst(rst), .start(mul_start), .a(mul_a), .b(mul_b),
        .busy(), .done(mul_done), .p(mul_p));

    reg          sq_start;
    reg  [143:0] sq_n;
    wire         sq_done;
    wire [71:0]  sq_root;
    isqrt_seq #(.WR(72)) u_sqrt (
        .clk(clk), .rst(rst), .start(sq_start), .n(sq_n),
        .busy(), .done(sq_done), .root(sq_root));

    // ------------------------------------------------------------ accumulators
    reg signed [63:0] p0 [0:8];
    reg signed [63:0] p1_dcz;
    reg signed [33:0] sbase;
    reg signed [25:0] m;
    reg signed [63:0] s_rms;
    reg signed [26:0] n2, p2;
    reg [10:0]        n2_idx;
    reg signed [39:0] s_n2m, s_p2m;
    reg signed [63:0] sx, sxx, sdd;
    reg signed [23:0] x500, x1499, xprev;
    reg signed [63:0] sa, sb, saa, sbb, sab;
    reg signed [71:0] hj_a, hj_b, cu, cv, cw, t1;
    reg [71:0]        cd;

    // ------------------------------------------------------------ control
    localparam S_IDLE = 4'd0, S_SET = 4'd1, S_W = 4'd2, S_USE = 4'd3, S_MW = 4'd4,
               S_ACC = 4'd5, S_NEXT = 4'd6, S_MEAN = 4'd7, S_MEANW = 4'd8,
               S_FIN = 4'd9, S_FINW = 4'd10, S_SQW = 4'd11, S_DONE = 4'd12;

    reg [3:0]  state;
    reg [3:0]  run_r;
    reg [1:0]  phase;
    reg [1:0]  step;                 // product index within one sample
    reg [10:0] j, j_end;
    reg [3:0]  fstep;
    reg signed [26:0] ep;            // current ERP sample minus baseline mean
    reg signed [23:0] va, vb;        // current samples of A and B

    wire signed [23:0] cur = (run_r == 4'd8 && phase == 2'd0) ? rb_data : ra_data;
    wire signed [26:0] cur_ep = $signed({{3{cur[23]}}, cur}) - $signed({m[25], m});
    wire is_pow  = (run_r != 4'd0) && (run_r != 4'd5) && !(run_r == 4'd8 && phase == 2'd1);
    wire is_hj   = (run_r == 4'd5);
    wire is_coup = (run_r == 4'd8) && (phase == 2'd1);

    function signed [71:0] ext64;
        input signed [63:0] v;
        ext64 = {{8{v[63]}}, v};
    endfunction

    // start a new pass over [first, last]
    task begin_pass;
        input [10:0] first;
        input [10:0] last;
        begin
            j     <= first;
            j_end <= last;
            state <= S_SET;
        end
    endtask

    always @(posedge clk) begin
        done      <= 1'b0;
        mul_start <= 1'b0;
        sq_start  <= 1'b0;
        if (rst) begin
            state <= S_IDLE;
        end else begin
            case (state)
            S_IDLE: begin
                if (scan_start) begin
                    run_r <= run;
                    phase <= 2'd0;
                    case (run)
                    4'd0: begin                       // ERP: baseline sum first
                        sbase  <= 34'sd0; s_rms <= 64'sd0; s_n2m <= 40'sd0; s_p2m <= 40'sd0;
                        begin_pass(11'd0, 11'd499);
                    end
                    4'd5: begin
                        sx <= 64'sd0; sxx <= 64'sd0; sdd <= 64'sd0;
                        begin_pass(11'd500, 11'd1499);
                    end
                    default: begin                    // power runs (and run 8 phase 0)
                        p0[run] <= 64'sd0;
                        if (run == 4'd1) p1_dcz <= 64'sd0;
                        if (run == 4'd8) begin
                            sa <= 64'sd0; sb <= 64'sd0; saa <= 64'sd0; sbb <= 64'sd0; sab <= 64'sd0;
                        end
                        // only delta Cz needs w1 (800..1099); the rest stop at w0
                        begin_pass(11'd500, (run == 4'd1) ? 11'd1099 : 11'd799);
                    end
                    endcase
                end else if (fin_start) begin
                    fstep <= 4'd0;
                    state <= S_FIN;
                end
            end

            S_SET: begin
                ra_addr <= j;
                rb_addr <= j;
                step    <= 2'd0;
                state   <= S_W;
            end
            S_W: state <= S_USE;

            S_USE: begin
                va <= ra_data;
                vb <= rb_data;
                if (run_r == 4'd0) begin
                    if (phase == 2'd0) begin
                        sbase <= sbase + cur;
                        state <= S_NEXT;
                    end else if (phase == 2'd1) begin
                        ep   <= cur_ep;
                        sm_a <= cur_ep;  sm_b <= cur_ep;
                        if (j >= 11'd650 && j < 11'd800 && (j == 11'd650 || cur_ep < n2)) begin
                            n2     <= cur_ep;
                            n2_idx <= j;
                        end
                        if (j >= 11'd650 && j < 11'd750) s_n2m <= s_n2m + cur_ep;
                        if (j >= 11'd750 && j < 11'd950) s_p2m <= s_p2m + cur_ep;
                        state <= S_MW;
                    end else begin                     // phase 2: P2 search
                        if (j == n2_idx + 11'd1 || cur_ep > p2) p2 <= cur_ep;
                        state <= S_NEXT;
                    end
                end else if (is_hj) begin
                    sx   <= sx + cur;
                    sm_a <= cur;  sm_b <= cur;
                    if (j == 11'd500)  x500  <= cur;
                    if (j == 11'd1499) x1499 <= cur;
                    state <= S_MW;
                end else if (is_coup) begin
                    sa   <= sa + ra_data;
                    sb   <= sb + rb_data;
                    sm_a <= ra_data;  sm_b <= ra_data;
                    state <= S_MW;
                end else begin                         // power
                    sm_a <= cur;  sm_b <= cur;
                    state <= S_MW;
                end
            end

            S_MW: state <= S_ACC;

            S_ACC: begin
                state <= S_NEXT;
                if (run_r == 4'd0) begin
                    s_rms <= s_rms + sq64;
                end else if (is_hj) begin
                    if (step == 2'd0) begin
                        sxx <= sxx + sq64;
                        if (j != 11'd500) begin        // first difference d = x - xprev
                            sm_a <= va - xprev;  sm_b <= va - xprev;
                            step <= 2'd1;
                            state <= S_MW;
                        end
                        xprev <= va;
                    end else begin
                        sdd <= sdd + sq64;
                    end
                end else if (is_coup) begin
                    case (step)
                    2'd0: begin saa <= saa + sq64; sm_a <= vb; sm_b <= vb; step <= 2'd1; state <= S_MW; end
                    2'd1: begin sbb <= sbb + sq64; sm_a <= va; sm_b <= vb; step <= 2'd2; state <= S_MW; end
                    default: sab <= sab + sq64;
                    endcase
                end else begin
                    if (j < 11'd800)          p0[run_r] <= p0[run_r] + sq64;
                    else if (run_r == 4'd1)   p1_dcz    <= p1_dcz + sq64;
                end
            end

            S_NEXT: begin
                if (j != j_end) begin
                    j     <= j + 11'd1;
                    state <= S_SET;
                end else if (run_r == 4'd0 && phase == 2'd0) begin
                    mul_a     <= {{38{sbase[33]}}, sbase};       // m = rnd(Sbase / 500)
                    mul_b     <= {40'd0, RECIP_500};
                    mul_start <= 1'b1;
                    state     <= S_MEANW;
                end else if (run_r == 4'd0 && phase == 2'd1) begin
                    phase <= 2'd2;
                    begin_pass(n2_idx + 11'd1, 11'd1049);
                end else if (run_r == 4'd8 && phase == 2'd0) begin
                    phase <= 2'd1;
                    begin_pass(11'd500, 11'd1499);
                end else begin
                    state <= S_DONE;
                end
            end

            S_MEANW: if (mul_done) begin
                m     <= (mul_p + 144'sd2147483648) >>> 32;
                phase <= 2'd1;
                begin_pass(11'd500, 11'd1099);
            end

            // ---------------- finalisation: 11 exact wide products, one square root
            S_FIN: begin
                case (fstep)
                4'd0:  begin mul_a <= 72'sd999;  mul_b <= ext64(sdd); end
                4'd1:  begin mul_a <= ext64(x1499 - x500); mul_b <= ext64(x1499 - x500); end
                4'd2:  begin mul_a <= 72'sd1000; mul_b <= ext64(sxx); end
                4'd3:  begin mul_a <= ext64(sx);  mul_b <= ext64(sx);  end
                4'd4:  begin mul_a <= 72'sd1000; mul_b <= ext64(sab); end
                4'd5:  begin mul_a <= ext64(sa);  mul_b <= ext64(sb);  end
                4'd6:  begin mul_a <= 72'sd1000; mul_b <= ext64(saa); end
                4'd7:  begin mul_a <= ext64(sa);  mul_b <= ext64(sa);  end
                4'd8:  begin mul_a <= 72'sd1000; mul_b <= ext64(sbb); end
                4'd9:  begin mul_a <= ext64(sb);  mul_b <= ext64(sb);  end
                default: begin mul_a <= cv;      mul_b <= cw;          end
                endcase
                mul_start <= 1'b1;
                state     <= S_FINW;
            end
            S_FINW: if (mul_done) begin
                case (fstep)
                4'd0, 4'd2, 4'd4, 4'd6, 4'd8: t1 <= mul_p[71:0];
                4'd1:  hj_a <= t1 - mul_p[71:0];
                4'd3:  hj_b <= t1 - mul_p[71:0];
                4'd5:  cu   <= t1 - mul_p[71:0];
                4'd7:  cv   <= t1 - mul_p[71:0];
                4'd9:  cw   <= t1 - mul_p[71:0];
                default: ;
                endcase
                if (fstep == 4'd10) begin
                    sq_n     <= mul_p;                 // v*w >= 0
                    sq_start <= 1'b1;
                    state    <= S_SQW;
                end else begin
                    fstep <= fstep + 4'd1;
                    state <= S_FIN;
                end
            end
            S_SQW: if (sq_done) begin
                cd    <= sq_root;
                state <= S_DONE;
            end

            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------ (A, B) read port
    localparam signed [71:0] K300 = 72'sd314572800;   // 300 * 2^20
    localparam signed [71:0] K600 = 72'sd629145600;   // 600 * 2^20
    localparam signed [71:0] K1   = 72'sd1024;

    always @(*) begin
        case (sel)
        4'd0:  begin a_out = ext64(p0[1]);     b_out = K300; end
        4'd1:  begin a_out = ext64(s_rms);     b_out = K600; end
        4'd2:  begin a_out = ext64(p0[2]);     b_out = K300; end
        4'd3:  begin a_out = cu;               b_out = $signed(cd); end
        4'd4:  begin a_out = {{45{p2[26]}}, p2};                b_out = K1; end
        4'd5:  begin a_out = {{45{p2[26]}}, p2} - {{45{n2[26]}}, n2}; b_out = K1; end
        4'd6:  begin a_out = {{45{n2[26]}}, n2};                b_out = K1; end
        4'd7:  begin a_out = ext64(p0[7]);     b_out = K300; end
        4'd8:  begin a_out = ext64(p0[1]);     b_out = ext64(p0[4]); end
        4'd9:  begin a_out = ext64(p1_dcz);    b_out = K300; end
        4'd10: begin a_out = ext64(p0[8]);     b_out = K300; end
        4'd11: begin a_out = ext64(p0[6]);     b_out = ext64(p0[8]); end
        4'd12: begin a_out = {{32{s_p2m[39]}}, s_p2m}; b_out = 72'sd204800; end
        4'd13: begin a_out = {{32{s_n2m[39]}}, s_n2m}; b_out = 72'sd102400; end
        4'd14: begin a_out = ext64(p0[3]);     b_out = ext64(p0[1]); end
        default: begin a_out = hj_a;           b_out = hj_b; end
        endcase
    end
endmodule

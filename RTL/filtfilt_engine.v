// Zero-phase band-pass filter engine: a bit-exact hardware twin of
// scipy.signal.sosfiltfilt(butter(4, band, 'bandpass', output='sos'), x) as used in
// training, including the 27-sample odd padding and the zi * x0 initial states.
//
// One run = one (band, channel) pair over the whole 3000-sample epoch:
//   forward pass   ext[0..3053] (padded raw epoch)  -> work RAM
//   backward pass  work RAM read in reverse         -> feature buffer
// Only output samples [onset-500, onset+1000) are written to the feature buffer
// (1500 samples): every feature window lies inside that span.
//
// Arithmetic (see scripts/hardware/hw_golden.py, which this matches bit for bit):
//   state / signal   signed 48 bit, 24 fractional bits (LSB 2^-24 uV)
//   coefficients     signed 36 bit, 32 fractional bits
//   biquad (DF2T)    y   = sat(z0 + rnd(b0*v))
//                    z0' = sat(z1 + rnd(b1*v - a1*y))
//                    z1' = sat(     rnd(b2*v - a2*y))
//   rnd(P) = (P + 2^31) >>> 32,  sat = clamp to 48 bits
//   output           sat24((y + 2^13) >>> 14)   (LSB 2^-10 uV)
//
// One shared 48x36 multiplier (2 pipeline registers, maps to DSP48s) is reused for
// every product. 41 cycles per sample per pass, ~250k cycles per run.
module filtfilt_engine #(
    parameter PRE       = 1000,          // onset position inside the epoch
    parameter COEF_FILE = "coef.mem"
) (
    input  wire               clk,
    input  wire               rst,
    input  wire               start,
    input  wire [2:0]         band,      // 0 delta 1 theta 2 beta 3 gamma 4 erp 5 broad
    input  wire [1:0]         ch,        // 0 Cz 1 C3 2 C4
    output reg  [11:0]        raw_idx,   // epoch-relative sample index requested
    input  wire [59:0]        raw_word,  // {C4, C3, Cz}; top registers the address,
                                         // so data arrives 2 cycles after raw_idx
    output reg                fb_we,
    output reg  [10:0]        fb_addr,
    output reg  signed [23:0] fb_data,
    output reg                done
);
    localparam N    = 3000;
    localparam PAD  = 27;
    localparam NEXT = N + 2 * PAD;       // 3054
    localparam LAST = NEXT - 1;          // 3053
    localparam FB_OFF = PAD + PRE - 500; // ext index of feature-buffer sample 0

    // ---------------------------------------------------------------- coefficients
    reg  [7:0]          crom_addr;
    wire [35:0]         crom_data;
    rom_sync #(.W(36), .AW(8), .FILE(COEF_FILE)) u_coef (
        .clk(clk), .addr(crom_addr), .data(crom_data));

    reg signed [35:0] cr [0:31];         // {sec, k}: b0 b1 b2 a1 a2 zi0 zi1 pad

    // ---------------------------------------------------------------- work RAM
    reg         w_we;
    reg  [11:0] w_waddr, w_raddr;
    reg  [47:0] w_wdata;
    wire [47:0] w_rdata;
    sdp_ram #(.W(48), .AW(12)) u_work (
        .clk(clk), .we(w_we), .waddr(w_waddr), .wdata(w_wdata),
        .raddr(w_raddr), .rdata(w_rdata));

    // ---------------------------------------------------------------- multiplier
    reg  signed [47:0] ma;
    reg  signed [35:0] mb;
    reg  signed [83:0] p1, p2;
    always @(posedge clk) begin
        p1 <= ma * mb;
        p2 <= p1;
    end

    function signed [55:0] rnd;          // (P + 2^31) >>> 32
        input signed [87:0] P;
        reg   signed [87:0] t;
        begin
            t   = P + 88'sd2147483648;
            rnd = t >>> 32;
        end
    endfunction

    function signed [47:0] sat48;
        input signed [57:0] v;
        begin
            if (v > 58'sd140737488355327)
                sat48 = 48'sh7fffffffffff;
            else if (v < -58'sd140737488355328)
                sat48 = 48'sh800000000000;
            else
                sat48 = v[47:0];
        end
    endfunction

    function signed [23:0] sat24;
        input signed [34:0] v;
        begin
            if (v > 35'sd8388607)
                sat24 = 24'sh7fffff;
            else if (v < -35'sd8388608)
                sat24 = 24'sh800000;
            else
                sat24 = v[23:0];
        end
    endfunction

    // ---------------------------------------------------------------- state
    localparam S_IDLE   = 5'd0,  S_LOAD  = 5'd1,  S_EDGE  = 5'd2,  S_FETCH = 5'd3,
               S_FW1    = 5'd4,  S_FW2   = 5'd5,  S_GOT   = 5'd6,  S_INIT  = 5'd7,
               S_INITC  = 5'd8,  S_SEC   = 5'd9,  S_Q1    = 5'd10, S_Q2    = 5'd11,
               S_Y      = 5'd12, S_Q4    = 5'd13, S_Q5    = 5'd14, S_Q6    = 5'd15,
               S_Z0     = 5'd16, S_Z1    = 5'd17, S_OUT   = 5'd18, S_MW1   = 5'd19,
               S_MW2    = 5'd20, S_DONE  = 5'd21;

    reg [4:0]  state, ret;
    reg [5:0]  lcnt;                     // coefficient preload / init counter
    reg        pass;                     // 0 forward, 1 backward
    reg [11:0] i;                        // ext index
    reg [1:0]  sec;
    reg [1:0]  ch_r;
    reg [2:0]  band_r;
    reg [1:0]  ecnt;
    reg signed [19:0] x_first, x_last;
    reg signed [47:0] v, y, x0;
    reg signed [47:0] z0 [0:3];
    reg signed [47:0] z1 [0:3];
    reg signed [83:0] acc1, acc2;

    wire signed [87:0] p2x   = {{4{p2[83]}}, p2};
    wire signed [87:0] acc1x = {{4{acc1[83]}}, acc1};
    wire signed [87:0] acc2x = {{4{acc2[83]}}, acc2};
    wire signed [55:0] r_p2  = rnd(p2x);             // rnd(b0*v), rnd(x0*zi)
    wire signed [55:0] r_a1  = rnd(acc1x - p2x);     // rnd(b1*v - a1*y)
    wire signed [55:0] r_a2  = rnd(acc2x - p2x);     // rnd(b2*v - a2*y)
    wire signed [47:0] z0s   = z0[sec];
    wire signed [47:0] z1s   = z1[sec];
    wire signed [57:0] y_sum  = $signed({{10{z0s[47]}}, z0s}) + $signed({{2{r_p2[55]}}, r_p2});
    wire signed [57:0] z0_sum = $signed({{10{z1s[47]}}, z1s}) + $signed({{2{r_a1[55]}}, r_a1});
    wire signed [57:0] z1_sum = $signed({{2{r_a2[55]}}, r_a2});
    wire signed [57:0] init_v = $signed({{2{r_p2[55]}}, r_p2});

    wire signed [19:0] raw_s = raw_word[ch_r*20 +: 20];
    wire signed [21:0] edge2 = (i < PAD) ? (2 * x_first) : (2 * x_last);

    // raw index for ext[i] (forward pass)
    function [11:0] ext_idx;
        input [11:0] k;
        begin
            if (k < PAD)            ext_idx = PAD - k;
            else if (k < PAD + N)   ext_idx = k - PAD;
            else                    ext_idx = 6025 - k;   // (N-2) - (k - (PAD+N))
        end
    endfunction

    always @(posedge clk) begin
        done  <= 1'b0;
        fb_we <= 1'b0;
        w_we  <= 1'b0;
        if (rst) begin
            state <= S_IDLE;
        end else begin
            case (state)
            S_IDLE: if (start) begin
                band_r    <= band;
                ch_r      <= ch;
                lcnt      <= 6'd0;
                crom_addr <= {band, 5'd0};
                state     <= S_LOAD;
            end

            // ROM latency is one cycle: address k is presented at edge k, the data
            // for address k is visible during the cycle after edge k+1.
            S_LOAD: begin
                crom_addr <= {band_r, lcnt[4:0] + 5'd1};
                if (lcnt >= 6'd1)
                    cr[lcnt - 6'd1] <= crom_data;
                if (lcnt == 6'd32) begin
                    raw_idx <= 12'd0;
                    ecnt    <= 2'd0;
                    state   <= S_EDGE;
                end
                lcnt <= lcnt + 6'd1;
            end

            // fetch x[0] and x[N-1] for the odd extension
            S_EDGE: begin
                ecnt <= ecnt + 2'd1;
                if (ecnt == 2'd0) raw_idx <= N - 1;
                if (ecnt == 2'd2) x_first <= raw_s;
                if (ecnt == 2'd3) begin
                    x_last <= raw_s;
                    pass   <= 1'b0;
                    i      <= 12'd0;
                    state  <= S_FETCH;
                end
            end

            S_FETCH: begin
                raw_idx <= ext_idx(i);
                w_raddr <= i;
                state   <= S_FW1;
            end
            S_FW1: state <= S_FW2;
            S_FW2: state <= S_GOT;

            S_GOT: begin
                if (!pass) begin
                    if (i < PAD || i >= PAD + N)
                        v <= $signed(edge2 - raw_s) <<< 14;
                    else
                        v <= $signed(raw_s) <<< 14;
                end else begin
                    v <= $signed(w_rdata);
                end
                sec <= 2'd0;
                if ((!pass && i == 0) || (pass && i == LAST)) begin
                    lcnt  <= 6'd0;
                    state <= S_INIT;
                end else begin
                    state <= S_SEC;
                end
            end

            // initial states: z[k][j] = sat(rnd(x0 * zi_j)), x0 = first sample of pass
            S_INIT: begin
                x0    <= v;
                ma    <= v;
                mb    <= cr[{lcnt[2:1], 3'd5 + lcnt[0]}];
                ret   <= S_INITC;
                state <= S_MW1;
            end
            S_INITC: begin
                if (lcnt[0]) z1[lcnt[2:1]] <= sat48(init_v);
                else         z0[lcnt[2:1]] <= sat48(init_v);
                if (lcnt == 6'd7) begin
                    state <= S_SEC;
                end else begin
                    lcnt  <= lcnt + 6'd1;
                    ma    <= x0;
                    mb    <= cr[{lcnt[2:1] + lcnt[0], 3'd5 + !lcnt[0]}];
                    ret   <= S_INITC;
                    state <= S_MW1;
                end
            end

            // One biquad section, input v. The multiplier has a 2-cycle pipeline, so
            // the five products are issued back to back where the data allows:
            //   Q0 v*b0  Q1 v*b1  Q2 v*b2  Q3 y = sat(z0 + rnd(v*b0))
            //   Q4 y*a1, acc1 = v*b1   Q5 y*a2, acc2 = v*b2   Q6 (wait)
            //   Q7 z0 = sat(z1 + rnd(acc1 - y*a1))   Q8 z1 = sat(rnd(acc2 - y*a2))
            S_SEC: begin ma <= v; mb <= cr[{sec, 3'd0}]; state <= S_Q1; end
            S_Q1:  begin ma <= v; mb <= cr[{sec, 3'd1}]; state <= S_Q2; end
            S_Q2:  begin ma <= v; mb <= cr[{sec, 3'd2}]; state <= S_Y;  end
            S_Y:   begin y <= sat48(y_sum); state <= S_Q4; end
            S_Q4:  begin acc1 <= p2; ma <= y; mb <= cr[{sec, 3'd3}]; state <= S_Q5; end
            S_Q5:  begin acc2 <= p2; ma <= y; mb <= cr[{sec, 3'd4}]; state <= S_Q6; end
            S_Q6:  state <= S_Z0;
            S_Z0:  begin z0[sec] <= sat48(z0_sum); state <= S_Z1; end
            S_Z1: begin
                z1[sec] <= sat48(z1_sum);
                v       <= y;                    // section output feeds the next one
                if (sec == 2'd3) begin
                    state <= S_OUT;
                end else begin
                    sec   <= sec + 2'd1;
                    state <= S_SEC;
                end
            end

            S_OUT: begin
                if (!pass) begin
                    w_we    <= 1'b1;
                    w_waddr <= i;
                    w_wdata <= v;
                    if (i == LAST) pass <= 1'b1;
                    else           i    <= i + 12'd1;
                    state <= S_FETCH;
                end else begin
                    if (i >= FB_OFF && i < FB_OFF + 1500) begin
                        fb_we   <= 1'b1;
                        fb_addr <= i - FB_OFF;
                        fb_data <= sat24(($signed({v[47], v}) + 49'sd8192) >>> 14);
                    end
                    if (i == 0) begin
                        state <= S_DONE;
                    end else begin
                        i     <= i - 12'd1;
                        state <= S_FETCH;
                    end
                end
            end

            S_MW1: state <= S_MW2;
            S_MW2: state <= ret;

            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule

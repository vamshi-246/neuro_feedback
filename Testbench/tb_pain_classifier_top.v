// System test: pain_classifier_top against the bit-exact golden model.
//
// For each epoch the testbench streams 3000 samples (trigger on sample 1000), then
// keeps streaming slow filler samples while the core processes -- as a real 1 kHz
// stream would -- and fires one extra trigger mid-processing, which must be dropped.
// It checks all 16 (A, B) feature pairs, all 16 8-bit codes, the 3 logits and the class.
//
// Vectors: py scripts/hardware/gen_test_vectors.py --out sim
// Run:     see sim/run_sim.sh      (+NEP=<n> limits the number of epochs)
`timescale 1ns/1ps
module tb_pain_classifier_top;
    localparam MAX_EP = 64;
    localparam GAP    = 2;          // idle cycles between streamed samples
    localparam FILL   = 20000;      // cycles between filler samples while busy

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg  [59:0] raw   [0:MAX_EP*3000-1];
    reg  [7:0]  eq    [0:MAX_EP*16-1];
    reg  [39:0] elog  [0:MAX_EP*3-1];
    reg  [3:0]  ecls  [0:MAX_EP-1];
    reg  [71:0] eab   [0:MAX_EP*32-1];

    reg               s_valid = 0, s_trigger = 0;
    reg  signed [19:0] s_cz = 0, s_c3 = 0, s_c4 = 0;
    wire              result_valid, busy;
    wire [1:0]        result_class;
    wire signed [39:0] logit_low, logit_mod, logit_high;
    wire [127:0]      feature_codes;
    wire [15:0]       triggers_dropped;

    pain_classifier_top #(.PRE(1000)) dut (
        .clk(clk), .rst(rst), .s_valid(s_valid), .s_cz(s_cz), .s_c3(s_c3), .s_c4(s_c4),
        .s_trigger(s_trigger), .result_valid(result_valid), .result_class(result_class),
        .logit_low(logit_low), .logit_mod(logit_mod), .logit_high(logit_high),
        .feature_codes(feature_codes), .busy(busy), .triggers_dropped(triggers_dropped));

    task send;
        input [59:0] w;
        input        trig;
        begin
            @(posedge clk);
            s_valid <= 1'b1;  s_trigger <= trig;
            s_cz <= w[19:0];  s_c3 <= w[39:20];  s_c4 <= w[59:40];
            @(posedge clk);
            s_valid <= 1'b0;  s_trigger <= 1'b0;
            repeat (GAP) @(posedge clk);
        end
    endtask

    integer nep, e0, e, n, i, errs, ep_errs, fill_n, dropped_before, t0;
    reg res_seen;
    integer cnt;
    always @(posedge clk) if (result_valid) res_seen = 1;
    reg [7:0] q;

    initial begin
        if (!$value$plusargs("NEP=%d", nep)) nep = 10;
        if (!$value$plusargs("E0=%d", e0)) e0 = 0;
        $readmemh("ep_raw.mem", raw);
        $readmemh("exp_q.mem", eq);
        $readmemh("exp_logit.mem", elog);
        $readmemh("exp_cls.mem", ecls);
        $readmemh("exp_ab.mem", eab);
        errs = 0;
        repeat (5) @(posedge clk);
        rst = 0;

        for (e = e0; e < e0 + nep; e = e + 1) begin
            ep_errs = 0;
            dropped_before = triggers_dropped;
            for (n = 0; n < 3000; n = n + 1)
                send(raw[e * 3000 + n], n == 1000);
            t0 = $time;
            // filler stream while busy; one stray trigger that must be dropped
            res_seen = 0;
            fill_n = 0;
            cnt = 0;
            while (!res_seen) begin
                @(posedge clk);
                cnt = cnt + 1;
                if (cnt == FILL && !res_seen) begin
                    send(60'd0, fill_n == 3);
                    fill_n = fill_n + 1;
                    cnt = 0;
                end
            end
            @(negedge clk);

            if (result_class !== ecls[e][1:0]) begin
                $display("  epoch %0d: class rtl %0d gold %0d", e, result_class, ecls[e]);
                ep_errs = ep_errs + 1;
            end
            if (logit_low !== elog[e*3] || logit_mod !== elog[e*3+1] || logit_high !== elog[e*3+2]) begin
                $display("  epoch %0d: logits rtl %h %h %h gold %h %h %h", e,
                         logit_low, logit_mod, logit_high, elog[e*3], elog[e*3+1], elog[e*3+2]);
                ep_errs = ep_errs + 1;
            end
            for (i = 0; i < 16; i = i + 1) begin
                q = feature_codes[8*i +: 8];
                if (q !== eq[e*16+i]) begin
                    $display("  epoch %0d: q[%0d] rtl %0d gold %0d", e, i, q, eq[e*16+i]);
                    ep_errs = ep_errs + 1;
                end
            end
            // (A, B) pairs, read through the feature engine's select port
            for (i = 0; i < 16; i = i + 1) begin
                force dut.u_fe.sel = i;
                #1;
                if (dut.u_fe.a_out !== eab[e*32+2*i] || dut.u_fe.b_out !== eab[e*32+2*i+1]) begin
                    $display("  epoch %0d: AB[%0d] rtl %h %h gold %h %h", e, i,
                             dut.u_fe.a_out, dut.u_fe.b_out, eab[e*32+2*i], eab[e*32+2*i+1]);
                    ep_errs = ep_errs + 1;
                end
            end
            release dut.u_fe.sel;
            if (fill_n > 3 && triggers_dropped !== dropped_before + 1) begin
                $display("  epoch %0d: stray trigger not counted (%0d -> %0d)", e,
                         dropped_before, triggers_dropped);
                ep_errs = ep_errs + 1;
            end
            $display("epoch %0d: class %0d, %0d mismatches, processing %0d cycles, %0d filler samples",
                     e, result_class, ep_errs, ($time - t0) / 10, fill_n);
            errs = errs + ep_errs;
        end
        if (errs == 0) $display("PASS pain_classifier_top: epochs %0d..%0d bit-exact", e0, e0 + nep - 1);
        else           $display("FAIL pain_classifier_top: %0d mismatches", errs);
        $finish;
    end
endmodule

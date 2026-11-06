// Unit test: filtfilt_engine against the golden feature buffers, every band x channel.
// Vectors: py scripts/hardware/gen_filter_unit.py sim
`timescale 1ns/1ps
module tb_filtfilt_engine;
    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg  [59:0] raw [0:2999];
    reg  [23:0] gold [0:26999];
    initial begin
        $readmemh("unit_raw.mem", raw);
        $readmemh("unit_fb.mem", gold);
    end

    reg         start = 0;
    reg  [2:0]  band;
    reg  [1:0]  ch;
    wire [11:0] raw_idx;
    reg  [11:0] raw_idx_q;
    reg  [59:0] raw_word;
    wire        fb_we, done;
    wire [10:0] fb_addr;
    wire signed [23:0] fb_data;

    // registered address, then registered data: 2-cycle latency, as in the top level
    always @(posedge clk) begin
        raw_idx_q <= raw_idx;
        raw_word  <= raw[raw_idx_q];
    end

    filtfilt_engine #(.PRE(1000), .COEF_FILE("coef.mem")) dut (
        .clk(clk), .rst(rst), .start(start), .band(band), .ch(ch),
        .raw_idx(raw_idx), .raw_word(raw_word),
        .fb_we(fb_we), .fb_addr(fb_addr), .fb_data(fb_data), .done(done));

    reg [23:0] got [0:1499];
    always @(posedge clk) if (fb_we) got[fb_addr] <= fb_data;

    integer b, c, j, errs, total, nb, b0;
    initial begin
        if (!$value$plusargs("NB=%d", nb)) nb = 6;
        if (!$value$plusargs("B0=%d", b0)) b0 = 0;
        total = 0;
        repeat (4) @(posedge clk);
        rst = 0;
        for (b = b0; b < b0 + nb; b = b + 1) begin
            for (c = 0; c < 3; c = c + 1) begin
                band = b; ch = c;
                @(posedge clk) start = 1;
                @(posedge clk) start = 0;
                @(posedge done);
                @(posedge clk);
                errs = 0;
                for (j = 0; j < 1500; j = j + 1)
                    if (got[j] !== gold[(b * 3 + c) * 1500 + j]) begin
                        if (errs < 3)
                            $display("  band %0d ch %0d j %0d: rtl %h gold %h", b, c, j,
                                     got[j], gold[(b * 3 + c) * 1500 + j]);
                        errs = errs + 1;
                    end
                $display("band %0d ch %0d: %0d mismatches (t=%0t)", b, c, errs, $time);
                $fflush;
                total = total + errs;
            end
        end
        if (total == 0) $display("PASS filtfilt_engine: %0d runs x 1500 samples bit-exact", nb * 3);
        else            $display("FAIL filtfilt_engine: %0d mismatches", total);
        $finish;
    end
endmodule

# Non-project Vivado synthesis of the no-MVAR pain classifier.
# Run from the RTL/ directory:
#     vivado -mode batch -source synth_vivado.tcl
# Default part is the one in the existing Vivado project (Zynq-7010). Override with
#     vivado -mode batch -source synth_vivado.tcl -tclargs <part>
set part xc7z010clg400-1
if {[llength $argv] > 0} { set part [lindex $argv 0] }

set here [file dirname [file normalize [info script]]]
cd $here

read_verilog [list \
    sdp_ram.v rom_sync.v mul_seq.v isqrt_seq.v \
    filtfilt_engine.v feature_engine.v threshold_quantizer.v linear_classifier.v \
    pain_classifier_top.v]
read_xdc pain_classifier.xdc

# ROM contents: absolute paths so $readmemh finds them wherever Vivado runs.
set m $here/mem
synth_design -top pain_classifier_top -part $part \
    -generic COEF_FILE=\"$m/coef.mem\" -generic TH_FILE=\"$m/thresh.mem\" \
    -generic W_FILE=\"$m/weights.mem\" -generic B_FILE=\"$m/bias.mem\"

report_utilization -file utilization_synth.rpt
report_timing_summary -file timing_synth.rpt
opt_design
place_design
route_design
report_utilization -file utilization_impl.rpt
report_timing_summary -file timing_impl.rpt
puts "done: see utilization_*.rpt and timing_*.rpt"

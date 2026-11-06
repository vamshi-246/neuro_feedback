# 50 MHz core clock. The design needs ~2.4 M cycles per epoch (~48 ms at 50 MHz)
# against a 2 s epoch, so there is no reason to push the clock; raise it only if the
# timing report has slack to spare. Pin assignments are board-specific and not set here.
create_clock -name clk -period 20.000 [get_ports clk]

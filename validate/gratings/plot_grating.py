# Licensed under a 3-clause BSD style license - see LICENSE.rst
msid_defs = """
4M5IRAX			ENAB		MCE A +5 VOLT CONV INH RELAY MONITOR
4MP5AV			~5		MCE A +5 VOLT MONITOR
4M5IRBX			ENAB		MCE B +5 VOLT CONV INH RELAY MONITOR
4MP5BV			~5		MCE B +5 VOLT MONITOR
4M28IRBX		ENAB		MCE B +28 VOLT CONV INH RELAY MONITOR
4MP28BV			~28		MCE B +28 VOLT MONITOR
4MP28AV			~6		MCE A +28 VOLT MONITOR

4HLORBX /		ENAB/DISA	MCE B HETG/LETG HARDWARE LOCKOUT 
4HRSMB  /	SELE/NSEL	MCE B HETG/LETG SELECT RELAY             
4HDIRB  /	INSR/RETR	MCE B HETG/LETG DIRECTION                
4HENLBX /		ENAB/DISA	MCE B HETG/LETG ENABLE LOGIC     
4HEXRBX /		ENAB/DISA	MCE B HETG/LETG EXECUTE          
4HILSB  /		INSR/NINS	MCE B HETG/LETG INSERT           
4HRLSB  /		RETR/NRET	MCE B HETG/LETG RETRACT          

4LLORBX /		ENAB/DISA	MCE B HETG/LETG HARDWARE LOCKOUT 
4LRSMB  /	SELE/NSEL	MCE B HETG/LETG SELECT RELAY              
4LDIRB  /	INSR/RETR	MCE B HETG/LETG DIRECTION                 
4LENLBX /		ENAB/DISA	MCE B HETG/LETG ENABLE LOGIC     
4LEXRBX /		ENAB/DISA	MCE B HETG/LETG EXECUTE          
4LILSBD /		INSR/NINS	MCE B HETG/LETG INSERT           
4LRLSBD /		RETR/NRET	MCE B HETG/LETG RETRACT          

4LPOSBRO
4LPOSARO
4HPOSBRO /	###		MCE B HETG/LETG Pot Angle
4HPOSARO /	###		MCE A HETG/LETG Pot Angle
4OOTGSEL	LETG/HETG	OTG SW GRATING SELECT
4OOTGMTN	INSE/RETR	OTG SW DIRECTION SELECT
4OOTGMEF	ENAB		OTG SW ENABLE MOTION
"""

msids = [x.split()[0] for x in msid_defs.strip().splitlines() if x]
if 'msidset' not in globals():
    # msidset = fetch.Msidset(msids, '2013:357:10:00:00', '2013:357:10:15:00')
    # msidset = fetch.Msidset(msids, '2000:001:10:00:00', '2000:010:10:15:00')
    # msidset = fetch.Msidset(msids, '2001:145:16:40:00', '2001:145:16:46:00')
    # msidset = fetch.Msidset(msids, '2001:188', '2001:191')
    msidset = fetch.Msidset(msids, '2000:048:08:09:19.544', '2000:048:08:14:36.344')


for ifig, i0, i1 in [(1, 0, 7), (2, 7, 14), (3, 14, 21), (4, 21, 28)]:
    figure(ifig)
    clf()
    for i_msid in range(i0, i1):
        msid = msidset[msids[i_msid]]
        i_sub = i_msid - i0 + 1
        subplot(i1 - i0, 1, i_sub)
        msid.plot('.-')
        grid('on')
        y0, y1 = ylim()
        dy = (y1 - y0) * 0.1
        ylim(y0 - dy, y1 + dy)
        ylabel(msids[i_msid])
    draw()
    show()

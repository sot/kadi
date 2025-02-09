# cd validate/pr343-maude-events

ln -s ../events/*.py ./

# Create kadi events in different configurations. The last process (test-maude) takes
# around 10 minutes to complete.

# Times spanning two safe mode transitions in 2023
START=2023:043
STOP_CREATE=2023:056
STOP_WRITE=2023:054

# Time spanning two eclipses and two SRDCs in 2024
START=2024:231
STOP_CREATE=2024:235
STOP_WRITE=2024:234

python create_events.py --code=flight --telem-source=cxc --start=$START --stop=${STOP_CREATE} | tee flight-cxc/run.log
python create_events.py --code=test --telem-source=cxc --start=$START --stop=${STOP_CREATE} | tee test-cxc/run.log
python create_events.py --code=test --telem-source=maude --start=$START --stop=${STOP_CREATE} | tee test-maude/run.log

python write_events_ecsv.py --data-root=flight-cxc --start=$START --stop=${STOP_WRITE}
python write_events_ecsv.py --data-root=test-cxc --start=$START --stop=${STOP_WRITE}
python write_events_ecsv.py --data-root=test-maude --start=$START --stop=${STOP_WRITE}
python write_events_ecsv.py --data-root=flight --start=$START --stop=${STOP_WRITE}

python compare_events_ecsv.py flight-cxc flight > flight-cxc_vs_flight.txt
python compare_events_ecsv.py flight-cxc test-cxc > flight-cxc_vs_test-cxc.txt
python compare_events_ecsv.py flight-cxc test-maude > flight-cxc_vs_test-maude.txt
python compare_events_ecsv.py test-cxc test-maude > test-cxc_vs_test-maude.txt

# For Safe Modes 2023
mv flight-cxc flight test-cxc test-maude *.txt safe-modes-2023/

# For Eclipses 2024
mv flight-cxc flight test-cxc test-maude *.txt eclipses-2024/

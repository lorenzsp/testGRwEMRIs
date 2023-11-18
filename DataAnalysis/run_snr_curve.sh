# Define the values for -M and devices
M_values=("100000.00000000001" "135935.6390878525" "184784.97974222922" "251188.643150958" "341454.88738336053" "464158.88336127804" "630957.3444801942" "857695.8985908948")
devices=("0" "1" "2" "3" "4" "5" "6" "7")

# Loop over each combination of -M and devices
for i in "${!M_values[@]}"; do
    M_value="${M_values[$i]}"
    device="${devices[$i]}"

    nohup python snr_curve.py -Tobs 2 -dt 10.0 -dev "$device" -M "$M_value" -mu 30.0 -e0 0.4 -a 0.98 -x0 1.0 > "snr_out${M_value}.out" &
done

M_values=("1165914.4011798317" "1584893.1924611153" "2154434.6900318847" "2928644.5646252357" "3981071.705534976" "5411695.265464638" "7356422.544596409" "10000000.000000006")
for i in "${!M_values[@]}"; do
    M_value="${M_values[$i]}"
    device="${devices[$i]}"

    nohup python snr_curve.py -Tobs 2 -dt 10.0 -dev "$device" -M "$M_value" -mu 30.0 -e0 0.4 -a 0.98 -x0 1.0 > "snr_out${M_value}.out" &
done
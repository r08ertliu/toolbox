#!/bin/bash

for p in $(lspci -d ::0108 -D -PP); do
	# Get NVMe root port BDF
	if ! [[ $p =~ ^([0-9a-f]+):(([0-9A-Fa-f:.]+)/)+ ]]; then
		continue
	fi
	domain=${BASH_REMATCH[1]}
	Root_Port_BDF=${domain}:${BASH_REMATCH[3]}
	slot_cap="0x$(setpci -s "$Root_Port_BDF" CAP_EXP+0x14.b)"
	indcator_cap=$(( (slot_cap & 0x18) >> 3 ))

	# Get NVMe device BDF
	if ! [[ ${p} =~ \/([0-9A-Fa-f:.]+)$ ]]; then
		continue
	fi

	if [[ ${indcator_cap} == 3 ]]; then
		echo "Indicator present CAP of slot ${Root_Port_BDF} of device ${domain}:${BASH_REMATCH[1]} is set"
	else
		echo "Indicator present CAP of slot ${Root_Port_BDF} of device ${domain}:${BASH_REMATCH[1]} does NOT set"
	fi
done

bash -lc 'set -euo pipefail
nettop -P -L 10 -m tcp | awk -F, '\''/^time/ {next} NF>=6 {proc=$2; sum=($5+0)+($6+0); if(!(proc in first)) first[proc]=sum; last[proc]=sum} END { for(p in last){d=last[p]-first[p]; if(d<0) d=0; printf "%12d %s\n", d, p } }'\'' | sort -nr | head -n 10'

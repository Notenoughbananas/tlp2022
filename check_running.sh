#!/bin/bash -e

destination=${1-uts}
echo "Check running of $destination"
if [ "$destination" == "uts"  ]; then
ssh jskardin@ihpc.eng.uts.edu.au 'cnode jskardin'
elif [ "$destination" == "abida"  ]; then
ssh $file asadaf@ihpc.eng.uts.edu.au 'cnode asadaf'
elif [ "$destination" == "aedan"  ]; then
ssh agrobert@ihpc.eng.uts.edu.au 'cnode agrobert'
else
  echo "Destination: $destination is not a valid destination"
fi


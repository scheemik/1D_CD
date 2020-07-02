#!/bin/bash
# A bash script to dump the structure (no data) of an h5 file
# Takes in arguments:
#	$ sh run.sh -i <path to h5file> <-- not optional
#				-o <output filepath>

while getopts i:o: option
do
	case "${option}"
		in
		i) IN_FILE=${OPTARG};;
		o) OUTFILE=${OPTARG};;
	esac
done

# check to see if arguments were passed
if [ -z "$IN_FILE" ]
then
	echo "-i, No input file specified specified, aborting script"
fi
if [ -z "$OUTFILE" ]
then
	OUTFILE="h5dump.txt"
	echo "-o, No output file specified, using $OUTFILE"
fi

###############################################################################

if [ -e $IN_FILE ]
then
	h5dump -A ${IN_FILE} > ${OUTFILE}
else
    echo "Input file ${IN_FILE} not found, aborting script"
fi

echo ''
echo "Done with h5 dump to ${OUTFILE}"
echo ''

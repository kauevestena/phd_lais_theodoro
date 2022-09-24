#!/bin/ksh

APP_PATH=/home/lais/Downloads/RTKLIB-master/app/rnx2rtkp/gcc/rnx2rtkp

OUT=/home/lais/Downloads/RTKLIB-master/app/rnx2rtkp/NAUS0101.pos

CONF=/media/lais/SAMSUNG/VERAO/NAUS/opts1.conf

RINEX=/media/lais/SAMSUNG/VERAO/NAUS/naus0011corr2.20o

RINEXNAV1=/media/lais/SAMSUNG/VERAO/NAUS/naus0011.20n

RINEXNAV2=/media/lais/SAMSUNG/VERAO/NAUS/naus0011.20g

sp3_1=/media/lais/SAMSUNG/VERAO/NAUS/igs20863.sp3

sp3_2=/media/lais/SAMSUNG/VERAO/NAUS/igl20863.sp3

clk=/media/lais/SAMSUNG/VERAO/NAUS/grg20863.clk

blqfile=/media/lais/SAMSUNG/VERAO/NAUS/naus.blq

satantfile=/media/lais/SAMSUNG/VERAO/AMCO/sat.atx

rcvantfile=/media/lais/SAMSUNG/VERAO/NAUS/naus.atx

dcbfile=/media/lais/SAMSUNG/VERAO/NAUS/P1C12001.DCB



chmod +x $APP_PATH

# echo $APP_PATH

#echo  $APP_PATH -lx -k $CONF -o $OUT $RINEX  #$RINEXNAV

echo $APP_PATH -lx -p 7 -k $CONF -o $OUT $RINEX $RINEXNAV1 $RINEXNAV2 $sp3_1 $sp3_2 $clk $blqfile $satantfile $rcvantfile $dcbfile

# rinex=/media/$USER/LAIS/laix/RINEX/braz1841_60.14o
# nav=/media/$USER/LAIS/laix/RINEX/braz1841.14n

rnx2rtkp -k ${CONF} -o ${OUT} ${RINEX} ${RINEXNAV1} ${RINEXNAV2} ${sp3_1} $sp3_2 ${clk} ${blqfile} ${satantfile} ${rcvantfile} ${dcbfile}

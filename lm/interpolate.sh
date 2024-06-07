BIN=/lfs01/workdirs/hlwn030u2/srilm/download/install/srilm-1.7.2/bin/i686-m64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lfs01/workdirs/hlwn030u2/srilm/download/install/liblbfgs-1.10/lib/.libs

ALPHA=0.1
					
echo =========ABSOLUTE DISCOUNTING FOURGRAM =========
$BIN/ngram -unk \
		-order $ORDER \
		-lm pure_mgb2.arpa \
		-mix-lm pure_140.arpa \
 		-lambda $ALPHA \
		-write-lm final${ALPHA}.arpa \
		

echo "DONE"

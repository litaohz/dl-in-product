#!/bin/bash
#sed 's/$1/$2/' dw_conf_2020_orig.ini
python spec_bucket.py  --conf dw_conf_2020_orig.ini 
cp entertainment_live_model_anchor_spec_bucket.ini entertainment_live_model_anchor_spec_bucket_$2.ini


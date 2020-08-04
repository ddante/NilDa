function O = Conv(X, K, sr, sc)

size(X);
size(K);

stride_r = sr;
stride_c = sc;

img_rows = size(X,1);
img_cols = size(X,2);

filter_rows = size(K,1);
filter_cols = size(K,2);

o_rows = 1 + (img_rows - filter_rows)/stride_r
o_cols = 1 + (img_cols - filter_cols)/stride_c

%=======================================

O = zeros(o_rows, o_cols);

idr_s = 1;
for i = 1:o_rows
  idr_e = idr_s + (filter_rows-1);
  
  idc_s = 1;  
  for j = 1:o_cols
    idc_e = idc_s + (filter_cols-1);
        
    O(i,j) = sum(sum( X(idr_s:idr_e, idc_s:idc_e).*K ));
        
    idc_s = idc_s + stride_c;
  end
  
  idr_s = idr_s + stride_r;
end
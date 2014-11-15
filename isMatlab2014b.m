function flag = isMatlab2014b()
[~, d] = version();
if datenum(d) >= datenum('September 15, 2014')
    flag = true;
else
    flag = false;
end
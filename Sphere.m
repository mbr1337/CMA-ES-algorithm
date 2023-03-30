function z = Sphere(x)

%     z = sum(x.^2); default
    d = length(x);
    sum = 0;
    prod = 1;
    
    for ii = 1:d
    	xi = x(ii);
    	sum = sum + xi^2/4000;
    	prod = prod * cos(xi/sqrt(ii));
    end
    
    z = sum + prod + 1;

end
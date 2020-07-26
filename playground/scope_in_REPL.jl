function foo();
    a = String[];
    for i = 1:5
        b = String[];
        for j=7:8
            b = vcat(b,"$j");    
        end
        a = vcat(a,b); # UndefVarError
    end
    println(a)
end

foo()


a = String[];
for i = 1:5
    b = String[];
    for j=7:8
        b = vcat(b,"$j");    
    end
    a = vcat(a,b); # UndefVarError
end
println(a)
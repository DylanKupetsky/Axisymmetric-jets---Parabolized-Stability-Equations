%Get the solution of the generalized eigenvalue problems
A = load('Abeta80c.mat','arrA_beta80c');
A = A.arrA_beta80c;
B = load('Bbeta80c.mat','arrB_beta80c');
B = B.arrB_beta80c;
[V80,D80] = eig(A,B,'qz');
[d80,ind80] = sort(diag(D80));

A = load('Abeta90c.mat','arrA_beta90c');
A = A.arrA_beta90c;
B = load('Bbeta90c.mat','arrB_beta90c');
B = B.arrB_beta90c;
[V90,D90] = eig(A,B,'qz');
[d90,ind90] = sort(diag(D90)); 

A = load('Abeta100c.mat','arrA_beta100c');
A = A.arrA_beta100c;
B = load('Bbeta100c.mat','arrB_beta100c');
B = B.arrB_beta100c;
[V100,D100] = eig(A,B,'qz');
[d100,ind100] = sort(diag(D100));

%Create a filter that returns 1 if the distances between eigenvalues is
%less than .8
mat1 = zeros(3*100,3*90);
mat2 = zeros(3*100,3*80);
for i = 1:3*100; for j = 1:3*90; mat1(i,j) = sqrt((real(d100(i)) - real(d90(j)))^2 + (imag(d100(i)) - imag(d90(j)))^2); end; end
for i = 1:3*100; for j = 1:3*80; mat2(i,j) = sqrt((real(d100(i)) - real(d80(j)))^2 + (imag(d100(i)) - imag(d80(j)))^2); end; end
[sm1r,sm1c] = find(mat1 < .3); 
[sm2r,sm2c] = find(mat2 < .3);
nodupes1 = unique(sm1r); 
nodupes2 = unique(sm2r);
nodupes = ismember(nodupes2,nodupes1); 
d100indices = linspace(1,3*100,3*100);
nodupesindices = nodupes2(nodupes);
d100truth = ismember(d100indices,nodupesindices);

%Creates a filter that returns 1 if the real part of the eigenvalue is
%between 0 and 1 and the imaginary between -1 and 0
d100im0 = imag(d100) < -1;
d100im1 = imag(d100) > -3;
d100re0 = real(d100) > 1; 
d100re1 = real(d100) < 3;
finallogical = d100im0.*d100im1.*d100re0.*d100re1; 

%Multiplies the two filters
conclusion = d100truth.*transpose(finallogical);
%If there is more than one good value of beta, delete all of the "1s" in
%the filter than come before the final value, ensuring that we get the
%largest imaginary part of beta
if sum(conclusion,'all') > 1
    tot = sum(conclusion,'all');
    a = find(conclusion == 1);
    for i = 1:tot - 1
        conclusion(a(i)) = 0;
    end
end
    

%Get the eigenvalue/eigenvector corresponding to this filter
beta = d100(logical(conclusion));
disp(beta);
Vtargetind = ind100(d100 == d100(logical(conclusion)));
Vtarget = V100(:,Vtargetind);
save('beta.mat','beta');
save('Vtarget.mat','Vtarget')
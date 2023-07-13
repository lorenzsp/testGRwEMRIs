(* ::Package:: *)

(* ::Title:: *)
(*ChebInt Package*)


(* ::Input::Initialization:: *)
BeginPackage["ChebInt`"]


(* ::Section:: *)
(*Preamble*)


ChebyshevNodes::usage=""


ChebyshevInterpolate::usage=""


(* ::Section:: *)
(*Private*)


(* ::Input::Initialization:: *)
Begin["`Private`"];


(* ::Subsection:: *)
(*ChebyshevNodes*)


ChebyshevNodes[n_,min_,max_]:=ChebyshevNodes[n,min,max,"Nodes"]


ChebyshevNodes[n_,min_,max_,"Nodes"]:=(max-min) (Table[-Cos[(2k-1)/(2n) \[Pi]],{k,1,n}]+1)/2+min


ChebyshevNodes[n_,min_,max_,"GL"]:=(max-min) (Table[-Cos[k/n \[Pi]],{k,0,n}]+1)/2+min


(* ::Subsection:: *)
(*1 D*)


ChebyshevInterpolate[data_,{nx_,xmin_,xmax_,type_:"Nodes"}]:=
Block[{Global`x},
Evaluate@Module[
{four,error},
four=FourierDCT[Reverse[Sort[data]][[All,2]]]/Sqrt[ nx] Array[If[#1==1,1,2]&,{nx}];
error=Max[Abs@{four[[-1]]}];
{
Function[{Global`x},Evaluate@Total[four Table[ChebyshevT[i,(Global`x-(xmin+xmax)/2)/((xmax-xmin)/2)],{i,0,nx-1}]]],
error,
four
}
]
]


ChebyshevInterpolate[data_,{nx_,xmin_,xmax_,"GL"}]:=
Block[{Global`x},
Evaluate@Module[
{four,error},
four=FourierDCT[Reverse[Sort[data]][[All,2]],1]/Sqrt[2 nx] Array[If[#1==1||#1==nx+1,1,2]&,{nx+1}];
error=Max[Abs@{four[[-1]]}];
{
Function[{Global`x},Evaluate@Total[four Table[ChebyshevT[i,(Global`x-(xmin+xmax)/2)/((xmax-xmin)/2)],{i,0,nx}]]],
error,
four
}
]
]


(* ::Subsection:: *)
(*2 D*)


ChebyshevInterpolate[data_,{nx_,xmin_,xmax_},{ny_,ymin_,ymax_}]:=
Block[{Global`x,Global`y},
Evaluate@Module[
{four,error},
four=1/Sqrt[ny nx] FourierDCT[Partition[(SortBy[{Minus@*N@*First,(-N@#[[2]]&)}]@data)[[All,3]],ny]]Array[If[#1==1,1,2]If[#2==1,1,2]&,{nx,ny}];
error=Max[Abs@{four[[-1]],four[[All,-1]]}];
{
Function[{Global`x,Global`y},Evaluate@Total[four Table[ChebyshevT[i,(Global`x-(xmin+xmax)/2)/((xmax-xmin)/2)]ChebyshevT[j,(Global`y-(ymin+ymax)/2)/((ymax-ymin)/2)],{i,0,nx-1},{j,0,ny-1}],2]],
error,
four
}
]
]


(* ::Subsection:: *)
(*3 D*)


ChebyshevInterpolate[data_,{nx_,xmin_,xmax_},{ny_,ymin_,ymax_},{nz_,zmin_,zmax_},type_:"Nodes"]:=
Block[{Global`x,Global`y,Global`z},
Evaluate@Module[
{four,error},
four=1/Sqrt[ny nx nz] FourierDCT[GatherBy[SortBy[Minus@*N]@data,{N@#[[1]]&,N@#[[2]]&}][[All,All,All,4]]]Array[If[#1==1,1,2]If[#2==1,1,2]If[#3==1,1,2]&,{nx,ny,nz}];
error=Max[Abs@{four[[-1]],four[[All,-1]],four[[All,All,-1]]}];
{
Function[{Global`x,Global`y,Global`z},Evaluate@Total[four Table[ChebyshevT[i,(Global`x-(xmin+xmax)/2)/((xmax-xmin)/2)]ChebyshevT[j,(Global`y-(ymin+ymax)/2)/((ymax-ymin)/2)]ChebyshevT[k,(Global`z-(zmin+zmax)/2)/((zmax-zmin)/2)],{i,0,nx-1},{j,0,ny-1},{k,0,nz-1}],3]],
error,
four
}
]
]


ChebyshevInterpolate[data_,{nx_,xmin_,xmax_},{ny_,ymin_,ymax_},{nz_,zmin_,zmax_},"GL"]:=
Block[{Global`x,Global`y,Global`z},
Evaluate@Module[
{four,error},
four=1/Sqrt[8ny nx nz] FourierDCT[GatherBy[SortBy[Minus@*N]@data,{N@#[[1]]&,N@#[[2]]&}][[All,All,All,4]],1]Array[If[#1==1||#1==nx+1,1,2]If[#2==1||#2==ny+1,1,2]If[#3==1||#3==nz+1,1,2]&,{nx+1,ny+1,nz+1}];
error=Max[Abs@{four[[-1]],four[[All,-1]],four[[All,All,-1]]}];
{
Function[{Global`x,Global`y,Global`z},Evaluate@Total[four Table[ChebyshevT[i,(Global`x-(xmin+xmax)/2)/((xmax-xmin)/2)]ChebyshevT[j,(Global`y-(ymin+ymax)/2)/((ymax-ymin)/2)]ChebyshevT[k,(Global`z-(zmin+zmax)/2)/((zmax-zmin)/2)],{i,0,nx},{j,0,ny},{k,0,nz}],3]],
error,
four
}
]
]


(* ::Section:: *)
(*End*)


(* ::Input::Initialization:: *)
End[];
EndPackage[];

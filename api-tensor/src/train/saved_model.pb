??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??
?
Adam/sequential/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/sequential/dense_2/bias/v
?
2Adam/sequential/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense_2/bias/v*
_output_shapes
:*
dtype0
?
 Adam/sequential/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*1
shared_name" Adam/sequential/dense_2/kernel/v
?
4Adam/sequential/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/sequential/dense_2/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/sequential/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/sequential/dense_1/bias/v
?
2Adam/sequential/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense_1/bias/v*
_output_shapes
:@*
dtype0
?
 Adam/sequential/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*1
shared_name" Adam/sequential/dense_1/kernel/v
?
4Adam/sequential/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/sequential/dense_1/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/sequential/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/sequential/dense/bias/v
?
0Adam/sequential/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/bias/v*
_output_shapes
:@*
dtype0
?
Adam/sequential/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name Adam/sequential/dense/kernel/v
?
2Adam/sequential/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/sequential/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/sequential/dense_2/bias/m
?
2Adam/sequential/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense_2/bias/m*
_output_shapes
:*
dtype0
?
 Adam/sequential/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*1
shared_name" Adam/sequential/dense_2/kernel/m
?
4Adam/sequential/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/sequential/dense_2/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/sequential/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/sequential/dense_1/bias/m
?
2Adam/sequential/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense_1/bias/m*
_output_shapes
:@*
dtype0
?
 Adam/sequential/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*1
shared_name" Adam/sequential/dense_1/kernel/m
?
4Adam/sequential/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/sequential/dense_1/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/sequential/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/sequential/dense/bias/m
?
0Adam/sequential/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/bias/m*
_output_shapes
:@*
dtype0
?
Adam/sequential/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name Adam/sequential/dense/kernel/m
?
2Adam/sequential/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/kernel/m*
_output_shapes

:@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
?
sequential/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential/dense_2/bias

+sequential/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential/dense_2/bias*
_output_shapes
:*
dtype0
?
sequential/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@**
shared_namesequential/dense_2/kernel
?
-sequential/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_2/kernel*
_output_shapes

:@*
dtype0
?
sequential/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namesequential/dense_1/bias

+sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential/dense_1/bias*
_output_shapes
:@*
dtype0
?
sequential/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@**
shared_namesequential/dense_1/kernel
?
-sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_1/kernel*
_output_shapes

:@@*
dtype0
?
sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_namesequential/dense/bias
{
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes
:@*
dtype0
?
sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_namesequential/dense/kernel
?
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel*
_output_shapes

:@*
dtype0

NoOpNoOp
?9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?9
value?9B?9 B?9
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
_build_input_shape

signatures*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_feature_columns

_resources* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator* 
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias*
.
0
1
-2
.3
54
65*
.
0
1
-2
.3
54
65*
* 
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
<trace_0
=trace_1
>trace_2
?trace_3* 
6
@trace_0
Atrace_1
Btrace_2
Ctrace_3* 
* 
?
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratem|m}-m~.m5m?6m?v?v?-v?.v?5v?6v?*
* 

Iserving_default* 
* 
* 
* 
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Otrace_0
Ptrace_1* 

Qtrace_0
Rtrace_1* 
* 
* 

0
1*

0
1*
* 
?
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
ga
VARIABLE_VALUEsequential/dense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEsequential/dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

_trace_0
`trace_1* 

atrace_0
btrace_1* 
* 

-0
.1*

-0
.1*
* 
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

htrace_0* 

itrace_0* 
ic
VARIABLE_VALUEsequential/dense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

50
61*

50
61*
* 
?
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
ic
VARIABLE_VALUEsequential/dense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

q0
r1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
s	variables
t	keras_api
	utotal
	vcount*
H
w	variables
x	keras_api
	ytotal
	zcount
{
_fn_kwargs*

u0
v1*

s	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

y0
z1*

w	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
??
VARIABLE_VALUEAdam/sequential/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/sequential/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/sequential/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/sequential/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/sequential/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/sequential/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/sequential/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/sequential/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/sequential/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/sequential/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/sequential/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/sequential/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
r
serving_default_label_0Placeholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
r
serving_default_label_1Placeholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_label_0serving_default_label_1sequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/bias*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_45170
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOp-sequential/dense_1/kernel/Read/ReadVariableOp+sequential/dense_1/bias/Read/ReadVariableOp-sequential/dense_2/kernel/Read/ReadVariableOp+sequential/dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2Adam/sequential/dense/kernel/m/Read/ReadVariableOp0Adam/sequential/dense/bias/m/Read/ReadVariableOp4Adam/sequential/dense_1/kernel/m/Read/ReadVariableOp2Adam/sequential/dense_1/bias/m/Read/ReadVariableOp4Adam/sequential/dense_2/kernel/m/Read/ReadVariableOp2Adam/sequential/dense_2/bias/m/Read/ReadVariableOp2Adam/sequential/dense/kernel/v/Read/ReadVariableOp0Adam/sequential/dense/bias/v/Read/ReadVariableOp4Adam/sequential/dense_1/kernel/v/Read/ReadVariableOp2Adam/sequential/dense_1/bias/v/Read/ReadVariableOp4Adam/sequential/dense_2/kernel/v/Read/ReadVariableOp2Adam/sequential/dense_2/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_45577
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/sequential/dense/kernel/mAdam/sequential/dense/bias/m Adam/sequential/dense_1/kernel/mAdam/sequential/dense_1/bias/m Adam/sequential/dense_2/kernel/mAdam/sequential/dense_2/bias/mAdam/sequential/dense/kernel/vAdam/sequential/dense/bias/v Adam/sequential/dense_1/kernel/vAdam/sequential/dense_1/bias/v Adam/sequential/dense_2/kernel/vAdam/sequential/dense_2/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_45668??
?

?
*__inference_sequential_layer_call_fn_45206
inputs_label_0	
inputs_label_1	
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_label_0inputs_label_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_45067o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:?????????
(
_user_specified_nameinputs/label_0:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/label_1
?:
?
E__inference_sequential_layer_call_and_return_conditional_losses_45257
inputs_label_0	
inputs_label_1	6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpp
%dense_features/label_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!dense_features/label_0/ExpandDims
ExpandDimsinputs_label_0.dense_features/label_0/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
dense_features/label_0/CastCast*dense_features/label_0/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????k
dense_features/label_0/ShapeShapedense_features/label_0/Cast:y:0*
T0*
_output_shapes
:t
*dense_features/label_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,dense_features/label_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,dense_features/label_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$dense_features/label_0/strided_sliceStridedSlice%dense_features/label_0/Shape:output:03dense_features/label_0/strided_slice/stack:output:05dense_features/label_0/strided_slice/stack_1:output:05dense_features/label_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&dense_features/label_0/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
$dense_features/label_0/Reshape/shapePack-dense_features/label_0/strided_slice:output:0/dense_features/label_0/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/label_0/ReshapeReshapedense_features/label_0/Cast:y:0-dense_features/label_0/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p
%dense_features/label_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!dense_features/label_1/ExpandDims
ExpandDimsinputs_label_1.dense_features/label_1/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
dense_features/label_1/CastCast*dense_features/label_1/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????k
dense_features/label_1/ShapeShapedense_features/label_1/Cast:y:0*
T0*
_output_shapes
:t
*dense_features/label_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,dense_features/label_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,dense_features/label_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$dense_features/label_1/strided_sliceStridedSlice%dense_features/label_1/Shape:output:03dense_features/label_1/strided_slice/stack:output:05dense_features/label_1/strided_slice/stack_1:output:05dense_features/label_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&dense_features/label_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
$dense_features/label_1/Reshape/shapePack-dense_features/label_1/strided_slice:output:0/dense_features/label_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/label_1/ReshapeReshapedense_features/label_1/Cast:y:0-dense_features/label_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/concatConcatV2'dense_features/label_0/Reshape:output:0'dense_features/label_1/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@h
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
#
_output_shapes
:?????????
(
_user_specified_nameinputs/label_0:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/label_1
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_45144
label_0	
label_1	
dense_45127:@
dense_45129:@
dense_1_45133:@@
dense_1_45135:@
dense_2_45138:@
dense_2_45140:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCalllabel_0label_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_45021?
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_45127dense_45129*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44866?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_44969?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_45133dense_1_45135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_44890?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_45138dense_2_45140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_44907w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	label_0:LH
#
_output_shapes
:?????????
!
_user_specified_name	label_1
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_44877

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_45394

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44866o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_44929
label_0	
label_1	
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllabel_0label_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_44914o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	label_0:LH
#
_output_shapes
:?????????
!
_user_specified_name	label_1
?	
?
*__inference_sequential_layer_call_fn_45100
label_0	
label_1	
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllabel_0label_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_45067o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	label_0:LH
#
_output_shapes
:?????????
!
_user_specified_name	label_1
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_45432

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?C
?
 __inference__wrapped_model_44815
label_0	
label_1	A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@C
1sequential_dense_1_matmul_readvariableop_resource:@@@
2sequential_dense_1_biasadd_readvariableop_resource:@C
1sequential_dense_2_matmul_readvariableop_resource:@@
2sequential_dense_2_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp{
0sequential/dense_features/label_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
,sequential/dense_features/label_0/ExpandDims
ExpandDimslabel_09sequential/dense_features/label_0/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
&sequential/dense_features/label_0/CastCast5sequential/dense_features/label_0/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:??????????
'sequential/dense_features/label_0/ShapeShape*sequential/dense_features/label_0/Cast:y:0*
T0*
_output_shapes
:
5sequential/dense_features/label_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7sequential/dense_features/label_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential/dense_features/label_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential/dense_features/label_0/strided_sliceStridedSlice0sequential/dense_features/label_0/Shape:output:0>sequential/dense_features/label_0/strided_slice/stack:output:0@sequential/dense_features/label_0/strided_slice/stack_1:output:0@sequential/dense_features/label_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1sequential/dense_features/label_0/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
/sequential/dense_features/label_0/Reshape/shapePack8sequential/dense_features/label_0/strided_slice:output:0:sequential/dense_features/label_0/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
)sequential/dense_features/label_0/ReshapeReshape*sequential/dense_features/label_0/Cast:y:08sequential/dense_features/label_0/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????{
0sequential/dense_features/label_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
,sequential/dense_features/label_1/ExpandDims
ExpandDimslabel_19sequential/dense_features/label_1/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
&sequential/dense_features/label_1/CastCast5sequential/dense_features/label_1/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:??????????
'sequential/dense_features/label_1/ShapeShape*sequential/dense_features/label_1/Cast:y:0*
T0*
_output_shapes
:
5sequential/dense_features/label_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7sequential/dense_features/label_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential/dense_features/label_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential/dense_features/label_1/strided_sliceStridedSlice0sequential/dense_features/label_1/Shape:output:0>sequential/dense_features/label_1/strided_slice/stack:output:0@sequential/dense_features/label_1/strided_slice/stack_1:output:0@sequential/dense_features/label_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1sequential/dense_features/label_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
/sequential/dense_features/label_1/Reshape/shapePack8sequential/dense_features/label_1/strided_slice:output:0:sequential/dense_features/label_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
)sequential/dense_features/label_1/ReshapeReshape*sequential/dense_features/label_1/Cast:y:08sequential/dense_features/label_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p
%sequential/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 sequential/dense_features/concatConcatV22sequential/dense_features/label_0/Reshape:output:02sequential/dense_features/label_1/Reshape:output:0.sequential/dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential/dense/MatMulMatMul)sequential/dense_features/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@~
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
sequential/dense_2/SigmoidSigmoid#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitysequential/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp:L H
#
_output_shapes
:?????????
!
_user_specified_name	label_0:LH
#
_output_shapes
:?????????
!
_user_specified_name	label_1
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_45472

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?A
?
E__inference_sequential_layer_call_and_return_conditional_losses_45315
inputs_label_0	
inputs_label_1	6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpp
%dense_features/label_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!dense_features/label_0/ExpandDims
ExpandDimsinputs_label_0.dense_features/label_0/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
dense_features/label_0/CastCast*dense_features/label_0/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????k
dense_features/label_0/ShapeShapedense_features/label_0/Cast:y:0*
T0*
_output_shapes
:t
*dense_features/label_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,dense_features/label_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,dense_features/label_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$dense_features/label_0/strided_sliceStridedSlice%dense_features/label_0/Shape:output:03dense_features/label_0/strided_slice/stack:output:05dense_features/label_0/strided_slice/stack_1:output:05dense_features/label_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&dense_features/label_0/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
$dense_features/label_0/Reshape/shapePack-dense_features/label_0/strided_slice:output:0/dense_features/label_0/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/label_0/ReshapeReshapedense_features/label_0/Cast:y:0-dense_features/label_0/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????p
%dense_features/label_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!dense_features/label_1/ExpandDims
ExpandDimsinputs_label_1.dense_features/label_1/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
dense_features/label_1/CastCast*dense_features/label_1/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????k
dense_features/label_1/ShapeShapedense_features/label_1/Cast:y:0*
T0*
_output_shapes
:t
*dense_features/label_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,dense_features/label_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,dense_features/label_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$dense_features/label_1/strided_sliceStridedSlice%dense_features/label_1/Shape:output:03dense_features/label_1/strided_slice/stack:output:05dense_features/label_1/strided_slice/stack_1:output:05dense_features/label_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&dense_features/label_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
$dense_features/label_1/Reshape/shapePack-dense_features/label_1/strided_slice:output:0/dense_features/label_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/label_1/ReshapeReshapedense_features/label_1/Cast:y:0-dense_features/label_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/concatConcatV2'dense_features/label_0/Reshape:output:0'dense_features/label_1/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
#
_output_shapes
:?????????
(
_user_specified_nameinputs/label_0:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/label_1
?
j
.__inference_dense_features_layer_call_fn_45327
features_label_0	
features_label_1	
identity?
PartitionedCallPartitionedCallfeatures_label_0features_label_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_45021`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????:?????????:U Q
#
_output_shapes
:?????????
*
_user_specified_namefeatures/label_0:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/label_1
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_44914

inputs	
inputs_1	
dense_44867:@
dense_44869:@
dense_1_44891:@@
dense_1_44893:@
dense_2_44908:@
dense_2_44910:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_44853?
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_44867dense_44869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44866?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_44877?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_44891dense_1_44893*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_44890?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_44908dense_2_44910*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_44907w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_45452

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_45067

inputs	
inputs_1	
dense_45050:@
dense_45052:@
dense_1_45056:@@
dense_1_45058:@
dense_2_45061:@
dense_2_45063:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_45021?
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_45050dense_45052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44866?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_44969?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_45056dense_1_45058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_44890?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_45061dense_2_45063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_44907w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_44866

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
.__inference_dense_features_layer_call_fn_45321
features_label_0	
features_label_1	
identity?
PartitionedCallPartitionedCallfeatures_label_0features_label_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_44853`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????:?????????:U Q
#
_output_shapes
:?????????
*
_user_specified_namefeatures/label_0:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/label_1
?	
?
#__inference_signature_wrapper_45170
label_0	
label_1	
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllabel_0label_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_44815o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	label_0:LH
#
_output_shapes
:?????????
!
_user_specified_name	label_1
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_44890

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_45420

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_45405

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
*__inference_sequential_layer_call_fn_45188
inputs_label_0	
inputs_label_1	
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_label_0inputs_label_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_44914o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:?????????
(
_user_specified_nameinputs/label_0:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/label_1
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_44907

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?o
?
!__inference__traced_restore_45668
file_prefix:
(assignvariableop_sequential_dense_kernel:@6
(assignvariableop_1_sequential_dense_bias:@>
,assignvariableop_2_sequential_dense_1_kernel:@@8
*assignvariableop_3_sequential_dense_1_bias:@>
,assignvariableop_4_sequential_dense_2_kernel:@8
*assignvariableop_5_sequential_dense_2_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: #
assignvariableop_13_total: #
assignvariableop_14_count: D
2assignvariableop_15_adam_sequential_dense_kernel_m:@>
0assignvariableop_16_adam_sequential_dense_bias_m:@F
4assignvariableop_17_adam_sequential_dense_1_kernel_m:@@@
2assignvariableop_18_adam_sequential_dense_1_bias_m:@F
4assignvariableop_19_adam_sequential_dense_2_kernel_m:@@
2assignvariableop_20_adam_sequential_dense_2_bias_m:D
2assignvariableop_21_adam_sequential_dense_kernel_v:@>
0assignvariableop_22_adam_sequential_dense_bias_v:@F
4assignvariableop_23_adam_sequential_dense_1_kernel_v:@@@
2assignvariableop_24_adam_sequential_dense_1_bias_v:@F
4assignvariableop_25_adam_sequential_dense_2_kernel_v:@@
2assignvariableop_26_adam_sequential_dense_2_bias_v:
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp(assignvariableop_sequential_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp(assignvariableop_1_sequential_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_sequential_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_sequential_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_sequential_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_sequential_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp2assignvariableop_15_adam_sequential_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_adam_sequential_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adam_sequential_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_sequential_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_sequential_dense_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_sequential_dense_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_sequential_dense_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_sequential_dense_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_sequential_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_sequential_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_sequential_dense_2_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp2assignvariableop_26_adam_sequential_dense_2_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
I__inference_dense_features_layer_call_and_return_conditional_losses_45356
features_label_0	
features_label_1	
identitya
label_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
label_0/ExpandDims
ExpandDimsfeatures_label_0label_0/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????r
label_0/CastCastlabel_0/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????M
label_0/ShapeShapelabel_0/Cast:y:0*
T0*
_output_shapes
:e
label_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
label_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
label_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
label_0/strided_sliceStridedSlicelabel_0/Shape:output:0$label_0/strided_slice/stack:output:0&label_0/strided_slice/stack_1:output:0&label_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
label_0/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
label_0/Reshape/shapePacklabel_0/strided_slice:output:0 label_0/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:~
label_0/ReshapeReshapelabel_0/Cast:y:0label_0/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????a
label_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
label_1/ExpandDims
ExpandDimsfeatures_label_1label_1/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????r
label_1/CastCastlabel_1/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????M
label_1/ShapeShapelabel_1/Cast:y:0*
T0*
_output_shapes
:e
label_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
label_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
label_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
label_1/strided_sliceStridedSlicelabel_1/Shape:output:0$label_1/strided_slice/stack:output:0&label_1/strided_slice/stack_1:output:0&label_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
label_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
label_1/Reshape/shapePacklabel_1/strided_slice:output:0 label_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:~
label_1/ReshapeReshapelabel_1/Cast:y:0label_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2label_0/Reshape:output:0label_1/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????:?????????:U Q
#
_output_shapes
:?????????
*
_user_specified_namefeatures/label_0:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/label_1
?
w
I__inference_dense_features_layer_call_and_return_conditional_losses_45021
features	

features_1	
identitya
label_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????}
label_0/ExpandDims
ExpandDimsfeatureslabel_0/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????r
label_0/CastCastlabel_0/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????M
label_0/ShapeShapelabel_0/Cast:y:0*
T0*
_output_shapes
:e
label_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
label_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
label_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
label_0/strided_sliceStridedSlicelabel_0/Shape:output:0$label_0/strided_slice/stack:output:0&label_0/strided_slice/stack_1:output:0&label_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
label_0/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
label_0/Reshape/shapePacklabel_0/strided_slice:output:0 label_0/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:~
label_0/ReshapeReshapelabel_0/Cast:y:0label_0/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????a
label_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
label_1/ExpandDims
ExpandDims
features_1label_1/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????r
label_1/CastCastlabel_1/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????M
label_1/ShapeShapelabel_1/Cast:y:0*
T0*
_output_shapes
:e
label_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
label_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
label_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
label_1/strided_sliceStridedSlicelabel_1/Shape:output:0$label_1/strided_slice/stack:output:0&label_1/strided_slice/stack_1:output:0&label_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
label_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
label_1/Reshape/shapePacklabel_1/strided_slice:output:0 label_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:~
label_1/ReshapeReshapelabel_1/Cast:y:0label_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2label_0/Reshape:output:0label_1/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????:?????????:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_44969

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_45415

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_44969o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
w
I__inference_dense_features_layer_call_and_return_conditional_losses_44853
features	

features_1	
identitya
label_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????}
label_0/ExpandDims
ExpandDimsfeatureslabel_0/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????r
label_0/CastCastlabel_0/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????M
label_0/ShapeShapelabel_0/Cast:y:0*
T0*
_output_shapes
:e
label_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
label_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
label_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
label_0/strided_sliceStridedSlicelabel_0/Shape:output:0$label_0/strided_slice/stack:output:0&label_0/strided_slice/stack_1:output:0&label_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
label_0/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
label_0/Reshape/shapePacklabel_0/strided_slice:output:0 label_0/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:~
label_0/ReshapeReshapelabel_0/Cast:y:0label_0/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????a
label_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
label_1/ExpandDims
ExpandDims
features_1label_1/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????r
label_1/CastCastlabel_1/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????M
label_1/ShapeShapelabel_1/Cast:y:0*
T0*
_output_shapes
:e
label_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
label_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
label_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
label_1/strided_sliceStridedSlicelabel_1/Shape:output:0$label_1/strided_slice/stack:output:0&label_1/strided_slice/stack_1:output:0&label_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
label_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
label_1/Reshape/shapePacklabel_1/strided_slice:output:0 label_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:~
label_1/ReshapeReshapelabel_1/Cast:y:0label_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2label_0/Reshape:output:0label_1/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????:?????????:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features
?
?
'__inference_dense_2_layer_call_fn_45461

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_44907o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
I__inference_dense_features_layer_call_and_return_conditional_losses_45385
features_label_0	
features_label_1	
identitya
label_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
label_0/ExpandDims
ExpandDimsfeatures_label_0label_0/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????r
label_0/CastCastlabel_0/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????M
label_0/ShapeShapelabel_0/Cast:y:0*
T0*
_output_shapes
:e
label_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
label_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
label_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
label_0/strided_sliceStridedSlicelabel_0/Shape:output:0$label_0/strided_slice/stack:output:0&label_0/strided_slice/stack_1:output:0&label_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
label_0/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
label_0/Reshape/shapePacklabel_0/strided_slice:output:0 label_0/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:~
label_0/ReshapeReshapelabel_0/Cast:y:0label_0/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????a
label_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
label_1/ExpandDims
ExpandDimsfeatures_label_1label_1/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????r
label_1/CastCastlabel_1/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????M
label_1/ShapeShapelabel_1/Cast:y:0*
T0*
_output_shapes
:e
label_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
label_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
label_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
label_1/strided_sliceStridedSlicelabel_1/Shape:output:0$label_1/strided_slice/stack:output:0&label_1/strided_slice/stack_1:output:0&label_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
label_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
label_1/Reshape/shapePacklabel_1/strided_slice:output:0 label_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:~
label_1/ReshapeReshapelabel_1/Cast:y:0label_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2label_0/Reshape:output:0label_1/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????:?????????:U Q
#
_output_shapes
:?????????
*
_user_specified_namefeatures/label_0:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/label_1
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_45122
label_0	
label_1	
dense_45105:@
dense_45107:@
dense_1_45111:@@
dense_1_45113:@
dense_2_45116:@
dense_2_45118:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense_features/PartitionedCallPartitionedCalllabel_0label_1*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_44853?
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_45105dense_45107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_44866?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_44877?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_45111dense_1_45113*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_44890?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_45116dense_2_45118*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_44907w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	label_0:LH
#
_output_shapes
:?????????
!
_user_specified_name	label_1
?
C
'__inference_dropout_layer_call_fn_45410

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_44877`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_45441

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_44890o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?>
?
__inference__traced_save_45577
file_prefix6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop8
4savev2_sequential_dense_1_kernel_read_readvariableop6
2savev2_sequential_dense_1_bias_read_readvariableop8
4savev2_sequential_dense_2_kernel_read_readvariableop6
2savev2_sequential_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_adam_sequential_dense_kernel_m_read_readvariableop;
7savev2_adam_sequential_dense_bias_m_read_readvariableop?
;savev2_adam_sequential_dense_1_kernel_m_read_readvariableop=
9savev2_adam_sequential_dense_1_bias_m_read_readvariableop?
;savev2_adam_sequential_dense_2_kernel_m_read_readvariableop=
9savev2_adam_sequential_dense_2_bias_m_read_readvariableop=
9savev2_adam_sequential_dense_kernel_v_read_readvariableop;
7savev2_adam_sequential_dense_bias_v_read_readvariableop?
;savev2_adam_sequential_dense_1_kernel_v_read_readvariableop=
9savev2_adam_sequential_dense_1_bias_v_read_readvariableop?
;savev2_adam_sequential_dense_2_kernel_v_read_readvariableop=
9savev2_adam_sequential_dense_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop4savev2_sequential_dense_1_kernel_read_readvariableop2savev2_sequential_dense_1_bias_read_readvariableop4savev2_sequential_dense_2_kernel_read_readvariableop2savev2_sequential_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_adam_sequential_dense_kernel_m_read_readvariableop7savev2_adam_sequential_dense_bias_m_read_readvariableop;savev2_adam_sequential_dense_1_kernel_m_read_readvariableop9savev2_adam_sequential_dense_1_bias_m_read_readvariableop;savev2_adam_sequential_dense_2_kernel_m_read_readvariableop9savev2_adam_sequential_dense_2_bias_m_read_readvariableop9savev2_adam_sequential_dense_kernel_v_read_readvariableop7savev2_adam_sequential_dense_bias_v_read_readvariableop;savev2_adam_sequential_dense_1_kernel_v_read_readvariableop9savev2_adam_sequential_dense_1_bias_v_read_readvariableop;savev2_adam_sequential_dense_2_kernel_v_read_readvariableop9savev2_adam_sequential_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:@:: : : : : : : : : :@:@:@@:@:@::@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
label_0,
serving_default_label_0:0	?????????
7
label_1,
serving_default_label_1:0	?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
_build_input_shape

signatures"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_feature_columns

_resources"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator"
_tf_keras_layer
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
J
0
1
-2
.3
54
65"
trackable_list_wrapper
J
0
1
-2
.3
54
65"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
<trace_0
=trace_1
>trace_2
?trace_32?
*__inference_sequential_layer_call_fn_44929
*__inference_sequential_layer_call_fn_45188
*__inference_sequential_layer_call_fn_45206
*__inference_sequential_layer_call_fn_45100?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z<trace_0z=trace_1z>trace_2z?trace_3
?
@trace_0
Atrace_1
Btrace_2
Ctrace_32?
E__inference_sequential_layer_call_and_return_conditional_losses_45257
E__inference_sequential_layer_call_and_return_conditional_losses_45315
E__inference_sequential_layer_call_and_return_conditional_losses_45122
E__inference_sequential_layer_call_and_return_conditional_losses_45144?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z@trace_0zAtrace_1zBtrace_2zCtrace_3
?B?
 __inference__wrapped_model_44815label_0label_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratem|m}-m~.m5m?6m?v?v?-v?.v?5v?6v?"
	optimizer
 "
trackable_dict_wrapper
,
Iserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Otrace_0
Ptrace_12?
.__inference_dense_features_layer_call_fn_45321
.__inference_dense_features_layer_call_fn_45327?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zOtrace_0zPtrace_1
?
Qtrace_0
Rtrace_12?
I__inference_dense_features_layer_call_and_return_conditional_losses_45356
I__inference_dense_features_layer_call_and_return_conditional_losses_45385?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zQtrace_0zRtrace_1
 "
trackable_list_wrapper
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Xtrace_02?
%__inference_dense_layer_call_fn_45394?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zXtrace_0
?
Ytrace_02?
@__inference_dense_layer_call_and_return_conditional_losses_45405?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zYtrace_0
):'@2sequential/dense/kernel
#:!@2sequential/dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
?
_trace_0
`trace_12?
'__inference_dropout_layer_call_fn_45410
'__inference_dropout_layer_call_fn_45415?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z_trace_0z`trace_1
?
atrace_0
btrace_12?
B__inference_dropout_layer_call_and_return_conditional_losses_45420
B__inference_dropout_layer_call_and_return_conditional_losses_45432?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zatrace_0zbtrace_1
"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
?
htrace_02?
'__inference_dense_1_layer_call_fn_45441?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zhtrace_0
?
itrace_02?
B__inference_dense_1_layer_call_and_return_conditional_losses_45452?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zitrace_0
+:)@@2sequential/dense_1/kernel
%:#@2sequential/dense_1/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
?
otrace_02?
'__inference_dense_2_layer_call_fn_45461?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zotrace_0
?
ptrace_02?
B__inference_dense_2_layer_call_and_return_conditional_losses_45472?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zptrace_0
+:)@2sequential/dense_2/kernel
%:#2sequential/dense_2/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_sequential_layer_call_fn_44929label_0label_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_45188inputs/label_0inputs/label_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_45206inputs/label_0inputs/label_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_45100label_0label_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_45257inputs/label_0inputs/label_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_45315inputs/label_0inputs/label_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_45122label_0label_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_45144label_0label_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
#__inference_signature_wrapper_45170label_0label_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_dense_features_layer_call_fn_45321features/label_0features/label_1"?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
.__inference_dense_features_layer_call_fn_45327features/label_0features/label_1"?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_dense_features_layer_call_and_return_conditional_losses_45356features/label_0features/label_1"?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
I__inference_dense_features_layer_call_and_return_conditional_losses_45385features/label_0features/label_1"?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_dense_layer_call_fn_45394inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_dense_layer_call_and_return_conditional_losses_45405inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dropout_layer_call_fn_45410inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_dropout_layer_call_fn_45415inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_45420inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_45432inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_1_layer_call_fn_45441inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_1_layer_call_and_return_conditional_losses_45452inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_2_layer_call_fn_45461inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_2_layer_call_and_return_conditional_losses_45472inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
N
s	variables
t	keras_api
	utotal
	vcount"
_tf_keras_metric
^
w	variables
x	keras_api
	ytotal
	zcount
{
_fn_kwargs"
_tf_keras_metric
.
u0
v1"
trackable_list_wrapper
-
s	variables"
_generic_user_object
:  (2total
:  (2count
.
y0
z1"
trackable_list_wrapper
-
w	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.:,@2Adam/sequential/dense/kernel/m
(:&@2Adam/sequential/dense/bias/m
0:.@@2 Adam/sequential/dense_1/kernel/m
*:(@2Adam/sequential/dense_1/bias/m
0:.@2 Adam/sequential/dense_2/kernel/m
*:(2Adam/sequential/dense_2/bias/m
.:,@2Adam/sequential/dense/kernel/v
(:&@2Adam/sequential/dense/bias/v
0:.@@2 Adam/sequential/dense_1/kernel/v
*:(@2Adam/sequential/dense_1/bias/v
0:.@2 Adam/sequential/dense_2/kernel/v
*:(2Adam/sequential/dense_2/bias/v?
 __inference__wrapped_model_44815?-.56f?c
\?Y
W?T
(
label_0?
label_0?????????	
(
label_1?
label_1?????????	
? "3?0
.
output_1"?
output_1??????????
B__inference_dense_1_layer_call_and_return_conditional_losses_45452\-./?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? z
'__inference_dense_1_layer_call_fn_45441O-./?,
%?"
 ?
inputs?????????@
? "??????????@?
B__inference_dense_2_layer_call_and_return_conditional_losses_45472\56/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? z
'__inference_dense_2_layer_call_fn_45461O56/?,
%?"
 ?
inputs?????????@
? "???????????
I__inference_dense_features_layer_call_and_return_conditional_losses_45356???}
v?s
i?f
1
label_0&?#
features/label_0?????????	
1
label_1&?#
features/label_1?????????	

 
p 
? "%?"
?
0?????????
? ?
I__inference_dense_features_layer_call_and_return_conditional_losses_45385???}
v?s
i?f
1
label_0&?#
features/label_0?????????	
1
label_1&?#
features/label_1?????????	

 
p
? "%?"
?
0?????????
? ?
.__inference_dense_features_layer_call_fn_45321???}
v?s
i?f
1
label_0&?#
features/label_0?????????	
1
label_1&?#
features/label_1?????????	

 
p 
? "???????????
.__inference_dense_features_layer_call_fn_45327???}
v?s
i?f
1
label_0&?#
features/label_0?????????	
1
label_1&?#
features/label_1?????????	

 
p
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_45405\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? x
%__inference_dense_layer_call_fn_45394O/?,
%?"
 ?
inputs?????????
? "??????????@?
B__inference_dropout_layer_call_and_return_conditional_losses_45420\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_45432\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? z
'__inference_dropout_layer_call_fn_45410O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@z
'__inference_dropout_layer_call_fn_45415O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
E__inference_sequential_layer_call_and_return_conditional_losses_45122?-.56n?k
d?a
W?T
(
label_0?
label_0?????????	
(
label_1?
label_1?????????	
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_45144?-.56n?k
d?a
W?T
(
label_0?
label_0?????????	
(
label_1?
label_1?????????	
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_45257?-.56|?y
r?o
e?b
/
label_0$?!
inputs/label_0?????????	
/
label_1$?!
inputs/label_1?????????	
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_45315?-.56|?y
r?o
e?b
/
label_0$?!
inputs/label_0?????????	
/
label_1$?!
inputs/label_1?????????	
p

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_44929?-.56n?k
d?a
W?T
(
label_0?
label_0?????????	
(
label_1?
label_1?????????	
p 

 
? "???????????
*__inference_sequential_layer_call_fn_45100?-.56n?k
d?a
W?T
(
label_0?
label_0?????????	
(
label_1?
label_1?????????	
p

 
? "???????????
*__inference_sequential_layer_call_fn_45188?-.56|?y
r?o
e?b
/
label_0$?!
inputs/label_0?????????	
/
label_1$?!
inputs/label_1?????????	
p 

 
? "???????????
*__inference_sequential_layer_call_fn_45206?-.56|?y
r?o
e?b
/
label_0$?!
inputs/label_0?????????	
/
label_1$?!
inputs/label_1?????????	
p

 
? "???????????
#__inference_signature_wrapper_45170?-.56a?^
? 
W?T
(
label_0?
label_0?????????	
(
label_1?
label_1?????????	"3?0
.
output_1"?
output_1?????????
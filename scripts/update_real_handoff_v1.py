"""Archive prior handoff and record the completed real-data work without changing old results."""
from pathlib import Path
import shutil

agent=Path('.agent');archive=agent/'archive';archive.mkdir(exist_ok=True)
for name in ['HANDOFF','TODO']:
    dest=archive/(name+'_before_real_cross_session_v1_2026-10-23.md')
    if dest.exists():raise FileExistsError(dest)
    shutil.copyfile(agent/(name+'.md'),dest)
handoff='''# Current objective

Real and simulated cross-session quantitative reinspection: physical identity plus valid material-anchor displacement, with acquisition costs. User explicitly requested actual public real cross-period data, published strong-method execution, causes of visible-but-unquantifiable objects, and next genuine algorithm innovation. No drone. Do not return to toy necessity checks or claim generic robust SE3/association/uncertainty is new.

# Latest completed 2026-10-23

real_cross_session_v1 COMPLETE. All jobs finished. Reports research/real_cross_session_v1_2026-10-23.md and research/rescan_author_execution_v1_2026-10-23.md. Public Rescan ICCV2019 official ZIP accessible, user has NO 3RScan authorization and explicitly chose openly downloadable alternatives. Never bypass 3RScan access. Six actual reconstructions: lexicographic scene_a/b/c captures0/1, selected before inference; 84,715,096 raw bytes, ~60.69MB transfer, selected ZIP CRC+SHA verified, full26GB ZIP not hashed. Raw RGBD streams NOT extracted. Stable instance IDs plus allowable ambiguity sets, NO independent object motion GT. Existing local Khronos labeled apartment/office are SYNTHETIC, OpenLORIS real57frames/36pairs lack motion/instance GT; do not relabel them.

# Real module experiment

configs/real_cross_session_v1.json frozen before inference. GT instance masks/classes privileged for this track; current persistent IDs randomized localtokens and withheld; predictions sealed before evaluator. H69 current69, 1current below128point excluded but retained inventory;68 eligible current,62 persistent+6new/missing-reference. 790 same-class candidatepairs x5methods=3950 fits,340 predictions,168.24sec, no failed fits. Shared geometry scoring and Hungarian/null adapter NOT original published end-to-end systems. centroid40/62 wrong4/unmatched18;centroidICP34 wrong27/unmatched1;FGR26 wrong27/unmatched9;FGR+GICP30 wrong29/unmatched3;TEASER27(24strict) wrong30/unmatched5. TEASER official f91cfdb7 native unchanged library, FPFH mutualmax256,noise.075; no certification invoked. FGR/GICP Open3D0.19 implementation. No same-FLOP/risk claim, centroid gate units differ. Geometry-only; ignores RGB and provided normals (reestimated).

POSTHOC ordinary RGB histogram controls no tuning: RGB30/62;RGB+position47 (wrong4/unmatched11);RGB+bestexistinggeometry35. Engineering gain, not new algorithm or independent confirmation.

Oracle strict-ID diagnostics62pairs:31with>=2 well-fitting estimates;only1 anchor displacement disagreement>.1m,4 nearbyvirtual-landmark>.1m. Not GT motion error or observability certificate. scene_b deskID6 has16.69cm ICP/FGR disagreement and nearplanar eigenratio.00645. Main failure is physicalID, not established widespread pose degeneracy. A wrong TEASER association scene_b true18->21 has98.66/95.50%support and~2cm median surface residual. Cannot use lowfitcost to certify physicalID. Global/background ICP also NOT cameraGT; scene_a refinement worsens sharedcost, not accepted.

# Actual full published baseline

Cloned original mhalber/Rescan and GradientSpaces/LivingScenes. OriginalRescan compiled butseg2rsdb exited11 all3. ASAN found64byteclassbuffer read512; isolated third_party/Rescan_memory_port_v1 fixes onlybuffers,initialization,filenameownership. 88otherfiles identical;onlyapps/seg2rsdb/main.cpp andlib/rs/rs_database.hchanged. Patch inresults/rescan_author_v1/memory_port/memory_only.patch. Singleportedseg2rsdb ASAN scene_c PASS, not globalmemorysafetyproof. Engineeringport not new algorithm.

Fullfirstupdate seg2rsdb->pose_proposal->segment_transfer COMPLETE scene_a16/20 (3wrongIDs1mixed)598.06s andscene_c14/14 277.79s. HistoricalGT initialization;current ALLIDs=-1/classes=0,otherfields EXACT. Completepipelineincludesnativeglobalarrangementgreedy/SA/refine/labeltransfer/database;postoutputPoissonfusion omitted, no nextperiod reconstruction claim. Eval>=80% currentGTpoints assigned allowedhistoricalID;not originalmIoU, no motionaccuracy. CoordinateorderEXACT,no NNmissing. a/b exceeded original240s candidatebudget;onlyfixedscene_a rerun900s completed;scene_b remains timeout NOT score0. Differentfront-end/compute fromoraclecurrentmask moduletrack: don'trank as fair/equalbudget.

ALL four native failures analyzed (POSTHOC evaluatoronly): ID7 raw11471 correctfraction.6897,44retainedproposals,bestcorrectsharedcost.327,H->Csupport.983/C->H.533 -> partialsupport/labeltransfer;ID12 raw10613 predicted10purity.9977,14proposals,bestcorrectcost.1225 support.962/.982 RMSE.0263 butfinaltrueIDtemplate~1.31mfromreal12;ID17 raw1900 predicted19all,26proposals,bestcorrectcost.329;ID23 raw1560 pred17purity.8115,34proposals,bestcorrectcost.548 support.642/.436. Correctcandidatecoverageexists12/17;newRANSACstarts notfirstissue. 7/23partialmodel/evidenceweakness notjustreordering. Parsedoriginalproposals.bin convention verified: int32narrays/counts,16columnmajorfloat+score. Shareddiagnosticcost NOT nativeglobalenergy, no global-optimum claim.

# Verification/runtime

4newunitPASS +1known-transform TEASERnativecheck(.000458m/.01245deg; syntheticUNITnotrealcase).8095mainchecksPASS incl22oldfrozensources unchanged;native88source/input/seal auditPASS. Plotscomparison.png,real_inputs.png andrescan_author_v1/author_comparison.png viewed. Finalindex records data/source/output hashes andrefs. Original UNION remainsREJECTED, old files unchanged.

Use .venv-real/Scripts/python.exe for newOpen3D scripts; independentvenv haspthtoexisting.venv numericalpackages, old.venv unchanged. PYTHONUTF8=1, OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1. TEASERnative requires WSLUbuntu20.04; sandboxWSL requires escalatedtool (autoapproved thisturn). GPUactualRTX5070TiLaptop12GB compute12.0;don'tassumeCUDA11.6/PyTorch1.13 MOREstack runs unmodified. LivingScenes official weights88,831,661bytes SHA48b86e3cda90a066287aa6ae2acf3b0c92416afc6288b1da6d25f0a370013231 present, inferenceNOTRUN. Originalrequirespy3.9/PyTorch1.13/PyTorch3D/torchlie/shapeextensions. ReScene4DCVPR2026 officialcodeexists butcheckpointREADMEcomingsoon;NOTRUN. C-3PO hascertifiablepartialpose priorart; confidence/observability notnew.

# Next actual algorithm work

Maincandidate: identity-and-material-displacement JOINT IDENTIFIABILITY under incomplete archive and globalframe uncertainty. Currentmath includesh=(A,G,D,S,V),jointP(correctID AND displacementerror<=epsilon),taskanchorDc-c,uncertaintysetUandChebyshevradius;finiteinitialization set is UNDERapprox, notposterior/certificate. Need computablecoveredouterbounds or calibratedjointreport mechanism, notjustconfidencewrapper. PriorartMOREjointmatch/register/reconstruct,ReScene4DtemporalID,C3POposecertification mustbecontrasted. NoveltyNOTCONFIRMED.

Next build concretecandidate and fulllearningstrongMOREsamecontract, usingnewindependentspaces ONLYafterfreezingcontract. Needindependentrealmaterial/poseGT toscoretruequantification;neveruseICP/self-consistencyasGT. Can usepublicrealobject-motioncalibrateddatasetseparatetrack, clearlysparsifiedvideonotcrossday. Mainwholemultiassettaskremains,don'tshrinktoonechair/desk. CompareordinaryRGB+position,multiinit/ensemblesusingEXACTsamehypothesiscalls,properselectivereportcontrol. Fixedcosttargetwrongreport<=5% withimprovedqualifiedcoverage, no rejectallgain. Ifstrongordinarymethodsolvescase,acceptengineeringinstead ofinventions.
'''
(agent/'HANDOFF.md').write_text(handoff,encoding='utf-8')
(agent/'TODO.md').write_text('''# Completed

Real Rescan3spaces/6captures;3950published-module fits,340sealed predictions;ordinaryRGBcontrols;actual full Rescan firstupdates a16/20,c14/14; btimeout notscored;four strong-baseline failures localized. All jobs ended. See HANDOFF and two new real_cross_session/rescan_author reports. Old UNIONclosed,old22frozenfiles unchanged.

# Next

- Implement a concrete identity/material-displacement JOINT candidate on the working real adapter, with explicit inference/bound novelty beyond generic EM/robustSE3/confidence.
- Deploy official LivingScenes code+presentpretrainedweight on same input contract; ReScene4Dcheckpointcomingsoon so don't invent comparisons.
- Add public real independent material-anchor/object-pose ground truth; current Rescan has persistent IDs but not independent motion GT. Do not use another ICP as truth.
- Freeze independent new scenes and compare fixed-budget qualified identity+motion coverage vs ordinaryRGB+position, publishedfullmethod, samehypothesisensemble/confidencecontrol. No retuning3alreadyobserved spaces as confirmation.
- Distinguish native failed initialization/memoryport/runtime limits (engineering) from incorrect identity despite retained good candidates (research target) and partialhistoricalgeometry (requires MORE comparison). Preserve all failures, do not revive old UNION.
''',encoding='utf-8')
summary='''2026-10-23 [真实跨期与强方法实验完成](research/real_cross_session_v1_2026-10-23.md)：取得Rescan三处真实空间六次重建，3950次已发表配准模块候选拟合、340个封存身份预测；普通RGB＋位置47/62。另[实际跑通Rescan作者完整首轮更新](research/rescan_author_execution_v1_2026-10-23.md)，当前ID／语义移除后scene_a16/20、scene_c14/14，scene_b超时不计分。全部四个强方法剩余失败已定位：两例良好正确身份候选存在但最终错分，两例部分历史表面／标签传播不足。原始崩溃、独立内存补丁、超时和负面结果保留。真实独立物体位移真值仍缺，论文创新尚未确认；下一项为部分档案下身份与材料位移的联合推断，不能再把普通稳健配准／融合颜色当创新。本段与最新HANDOFF优先于历史下一步。'''
p=Path('README.md');s=p.read_text(encoding='utf-8');head,tail=s.split('\n',1);p.write_text(head+'\n\n'+summary+'\n'+tail,encoding='utf-8')
for name in ['PROGRESS','DECISIONS','MAINLINE_AUDIT','CONTEXT']:
    p=agent/(name+'.md')
    if p.exists():p.write_text('## 2026-10-23 real cross-session execution completed\n\n'+summary+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
print('Handoff updated; prior handoff archived')

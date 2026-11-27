"use strict";
const $ = id => document.getElementById(id);
let dataset = null, inspection = null, analysis = null, submitted = null, busy = false;
const fmt = n => n == null ? "Unavailable" : Number(n).toFixed(4);
function notice(message, error = false) { $("notice").textContent = message; $("notice").className = `notice ${error ? "error" : "info"}`; $("notice").hidden = !message; }
function syncNavigation() {
  document.querySelectorAll("nav button").forEach(b => { b.disabled = busy || (b.dataset.page !== "dataset" && !inspection) || (["results", "report"].includes(b.dataset.page) && !analysis); });
  $("clear").disabled = busy || !inspection;
}
function page(name) {
  document.querySelectorAll(".page").forEach(el => { el.hidden = el.id !== name; });
  document.querySelectorAll("nav button").forEach(el => { if (el.dataset.page === name) el.setAttribute("aria-current", "page"); else el.removeAttribute("aria-current"); });
  document.querySelectorAll(".steps li").forEach(el => { if (el.dataset.step === name) el.setAttribute("aria-current", "step"); else el.removeAttribute("aria-current"); });
  notice(""); document.querySelector(`#${name} h2`).focus();
  if (name === "results") window.dispatchEvent(new Event("resize"));
}
async function task(message, action) {
  if (busy) return;
  busy = true; notice(message);
  document.querySelectorAll("button,input,select").forEach(el => { el.disabled = true; });
  try { await action(); } catch (e) { notice(e.message || "The request failed. Please try again.", true); }
  finally { busy = false; document.querySelectorAll("button,input,select").forEach(el => { el.disabled = false; }); syncNavigation(); }
}
async function api(path, data, pdf = false) {
  let response;
  try { response = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data), cache:"no-store"}); }
  catch { throw new Error("Could not reach the server. Check your connection and try again."); }
  if (!response.ok) { let msg = "The server could not complete the request. Please try again."; try { msg = (await response.json()).error || msg; } catch {} throw new Error(msg); }
  return pdf ? response.blob() : response.json();
}
function reset() {
  dataset = inspection = analysis = submitted = null;
  $("dataset-summary").hidden = true; $("dataset-name").textContent = "No dataset selected"; $("session-state").textContent = "Choose a sample or upload a CSV to begin.";
  $("file").value = ""; $("analysis-form").reset();
  ["selection-chart", "error-chart", "metrics", "error-metrics", "decisions"].forEach(id => $(id).replaceChildren());
  $("group-table").querySelector("tbody").replaceChildren(); $("group-table").querySelector("thead").replaceChildren(); syncNavigation();
}
function options(id, values, selected, optional = false) {
  $(id).replaceChildren();
  if (optional) $(id).add(new Option("No true outcome", ""));
  values.forEach(value => $(id).add(new Option(value, value)));
  if (selected != null && [...$(id).options].some(o => o.value === selected)) $(id).value = selected;
}
function values(column) { return inspection.columns.find(c => c.name === column)?.values || []; }
function labelOptions(positive, negative, column) {
  const vals = values(column);
  options(positive, vals, vals.includes("1") ? "1" : vals.includes(">50K") ? ">50K" : vals.at(-1));
  options(negative, vals.filter(v => v !== $(positive).value), vals.includes("0") ? "0" : vals[0]);
}
function updateMappings() {
  labelOptions("positive", "negative", $("prediction").value);
  labelOptions("true-positive", "true-negative", $("truth").value);
  $("truth-mapping").hidden = !$("truth").value;
  $("prediction-mapping").hidden = $("score-mode").checked;
  $("score-field").hidden = !$("score-mode").checked;
}
function invalidate() { analysis = submitted = null; $("session-state").textContent = "Ready to configure"; syncNavigation(); }
async function load(input) {
  const info = await api("/api/inspect", input);
  dataset = input; inspection = info; analysis = submitted = null;
  $("dataset-name").textContent = info.metadata.name; $("session-state").textContent = "Ready to configure";
  $("facts").replaceChildren();
  for (const [n, label] of [[info.rows,"rows"],[info.columns.length,"columns"],[info.columns.reduce((n,c) => n+c.missing,0),"missing values"]]) {
    const span = document.createElement("span"), strong = document.createElement("strong"); strong.textContent = n.toLocaleString(); span.append(strong, ` ${label}`); $("facts").append(span);
  }
  $("provenance").textContent = `Source: ${info.metadata.source}. Positive class means ${info.metadata.positive}.`;
  const missing = info.columns.filter(c => c.missing).map(c => `${c.name}: ${c.missing.toLocaleString()}`);
  $("missing").hidden = !missing.length; $("missing").textContent = `Missing values — ${missing.join("; ")}. Resolve missing values in selected fields before analysis.`;
  const columns = info.columns.map(c => c.name);
  const pred = ["risk_score","income_prediction","prediction"].find(c => columns.includes(c)) || columns[1];
  const group = ["race","group","sex"].find(c => columns.includes(c) && c !== pred) || columns.find(c => c !== pred);
  options("group", columns, group); options("prediction", columns, pred);
  options("truth", columns, ["recidivism_2yr","income_true","truth"].find(c => columns.includes(c)) || "", true);
  $("analysis-form").reset();
  $("group").value = group; $("prediction").value = pred;
  $("truth").value = ["recidivism_2yr","income_true","truth"].find(c => columns.includes(c)) || "";
  $("score-mode").checked = pred === "risk_score";
  updateMappings(); $("dataset-summary").hidden = false; syncNavigation(); notice("Dataset loaded. Continue to choose your comparison.");
}
function settings() {
  return {group_column:$("group").value,prediction_column:$("prediction").value,true_column:$("truth").value || null,score_mode:$("score-mode").checked,score_threshold:Number($("threshold").value),positive_label:$("score-mode").checked ? "1" : $("positive").value,negative_label:$("score-mode").checked ? "0" : $("negative").value,true_positive_label:$("truth").value ? $("true-positive").value : null,true_negative_label:$("truth").value ? $("true-negative").value : null,dpd_threshold:Number($("dpd").value),dir_threshold:Number($("dir").value),minimum_group_size:Number($("minimum").value)};
}
function metricList(id, pairs) {
  $(id).replaceChildren(); pairs.forEach(([name,value]) => { const div=document.createElement("div"),dt=document.createElement("dt"),dd=document.createElement("dd"); dt.textContent=name;dd.textContent=value;div.append(dt,dd);$(id).append(div); });
}
async function renderResults() {
  const {result, metadata} = analysis, m=result.metrics, groups=result.groups;
  $("positive-meaning").textContent = `Positive class: ${metadata.positive}.${metadata.adverse ? " This is an adverse risk flag, not a favourable selection." : ""}`;
  metricList("metrics", [["Rows analysed",m.rows_analysed.toLocaleString()],["All-group DPD (descriptive)",fmt(m.demographic_parity_difference)],["All-group DIR (descriptive)",fmt(m.disparate_impact_ratio)],["Eligible-group DPD",fmt(m.eligible_screening.dpd)],["Eligible-group DIR",fmt(m.eligible_screening.dir)],["Minimum group",m.settings.minimum_group_size]]);
  $("decisions").replaceChildren();
  const e=m.eligible_screening; const messages=[e.message, ...(e.excluded_groups.length ? ["Insufficient group data: "+e.excluded_groups.join(", ")] : [])];
  
  messages.forEach(message => {const p=document.createElement("p"); p.textContent=message; if(!m.reliable || m.demographic_parity_flag || m.disparate_impact_flag) p.className="warning"; $("decisions").append(p);});
  const keys=["group","n","sample_status","selection_rate",...(m.true_positive_rate !== null ? ["tpr","fpr","fnr"]:[])];
  const headings={group:"Group",n:"Rows",sample_status:"Sample status",selection_rate:"Selection rate",tpr:"TPR",fpr:"FPR",fnr:"FNR"};
  const table=$("group-table"); table.querySelector("thead").replaceChildren();table.querySelector("tbody").replaceChildren();
  const tr=document.createElement("tr");keys.forEach(key=>{const th=document.createElement("th");th.scope="col";th.textContent=headings[key];tr.append(th);});table.querySelector("thead").append(tr);
  groups.forEach(group=>{const tr=document.createElement("tr");keys.forEach(key=>{const td=document.createElement("td");td.textContent=["group","n","sample_status"].includes(key)?group[key]:fmt(group[key]);tr.append(td);});table.querySelector("tbody").append(tr);});
  const comp=m.selection_rate_comparison;
  const pFmt=v=>v===null || v===undefined ? "Unavailable" : (v===0 ? "Below numerical precision" : v.toExponential(4));
  $("statistic").textContent=`All-group descriptive extremes: ${comp.high_group} vs ${comp.low_group}. `+
    (e.high_group ? `Eligible-group extremes: ${e.high_group} vs ${e.low_group}.` : "No eligible comparison.");
  const details=document.createElement("details"); const summary=document.createElement("summary");
  summary.textContent="Eligible pairwise statistical comparisons"; details.append(summary);
  const pairTable=document.createElement("table"), head=document.createElement("tr");
  ["Higher rate","Lower rate","DPD","DIR","Z","Raw p","Adjusted p","Interpretation"].forEach(v=>{const th=document.createElement("th");th.scope="col";th.textContent=v;head.append(th);});
  const thead=document.createElement("thead");thead.append(head);pairTable.append(thead);
  const tbody=document.createElement("tbody");e.pairs.forEach(r=>{const tr=document.createElement("tr");[r.high_group,r.low_group,fmt(r.dpd),fmt(r.dir),fmt(r.z),pFmt(r.p),pFmt(r.adjusted_p),r.status].forEach(v=>{const td=document.createElement("td");td.textContent=v;tr.append(td);});tbody.append(tr);});pairTable.append(tbody);details.append(pairTable);$("decisions").append(details);
  $("undefined").hidden=!m.undefined.length;$("undefined").textContent=m.undefined.join(" ");
  $("error-section").hidden=m.true_positive_rate===null;
  metricList("error-metrics", [["Equal opportunity difference",fmt(m.equal_opportunity_difference)],["Equalised odds difference",fmt(m.equalised_odds_difference)]]);
  page("results");
  if(!window.Plotly) {notice("Charts could not load. All numerical results are available in the table. Refresh to retry.",true);return;}
  // Treat group labels as plain categories, not Plotly's HTML-like label markup.
  const escape=s=>String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
  const names=groups.map(g=>escape(g.group));
  const layout={height:360,margin:{l:52,r:20,t:20,b:85},font:{family:"system-ui",color:"#18352f"},yaxis:{tickformat:".0%",range:[0,1]},xaxis:{type:"category",automargin:true},paper_bgcolor:"#ffffff",plot_bgcolor:"#ffffff"};
  await Plotly.newPlot("selection-chart",[{type:"bar",x:names,y:groups.map(g=>g.selection_rate),marker:{color:"#176b55"},text:groups.map(g=>`${(g.selection_rate*100).toFixed(1)}%`),textposition:"auto",hovertemplate:"%{x}: %{y:.1%}<extra></extra>"}],layout,{responsive:true,displaylogo:false});
  if(m.true_positive_rate!==null) await Plotly.newPlot("error-chart",[...["tpr","fpr","fnr"].map((key,i)=>({type:"bar",name:headings[key],x:names,y:groups.map(g=>g[key]),marker:{color:["#176b55","#b35e30","#667a70"][i]}}))],{...layout,barmode:"group",legend:{orientation:"h",y:1.15}},{responsive:true,displaylogo:false});
}
document.querySelectorAll("nav button").forEach(b=>b.addEventListener("click",()=>page(b.dataset.page)));
document.querySelectorAll("input[name=source]").forEach(r=>r.addEventListener("change",()=>{reset();$("sample-fields").hidden=r.value!=="sample";$("upload-fields").hidden=r.value!=="upload";notice("");}));
$("clear").addEventListener("click",()=>{reset();page("dataset");});
$("sample").addEventListener("change",()=>{reset();notice("Load the selected sample to continue.");});
$("load-sample").addEventListener("click",()=>task("Loading sample…",()=>load({sample:$("sample").value})));
$("file").addEventListener("change",()=>{
 const file=$("file").files[0]; reset();
 if(!file)return;
 if(file.size>3*1024*1024){notice("Choose a CSV smaller than 3 MiB.",true);return;}
 task("Reading CSV in memory…",async()=>{const csv=new TextDecoder("utf-8",{fatal:true}).decode(await file.arrayBuffer());await load({csv});});
});
$("continue").addEventListener("click",()=>page("analysis"));
$("analysis-form").addEventListener("change",event=>{invalidate();if(["prediction","truth","score-mode"].includes(event.target.id))updateMappings();if(event.target.id==="positive")options("negative",values($("prediction").value).filter(v=>v!==$("positive").value));if(event.target.id==="true-positive")options("true-negative",values($("truth").value).filter(v=>v!==$("true-positive").value));});
$("analysis-form").addEventListener("submit",event=>{event.preventDefault();task("Calculating metrics and reliability checks…",async()=>{const config=settings(); const result=await api("/api/analyze",{...dataset,settings:config});analysis=result;submitted=config;$("session-state").textContent="Results ready";await renderResults();});});
$("prepare-report").addEventListener("click",()=>page("report"));
$("download").addEventListener("click",()=>task("Preparing your PDF…",async()=>{const blob=await api("/api/report",{...dataset,settings:submitted},true),url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download="bias_audit_summary.pdf";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notice("Report downloaded.");}));
const chartObserver = new ResizeObserver(entries => entries.forEach(({target}) => { if(window.Plotly && target.data && target.offsetWidth) Plotly.Plots.resize(target); }));
["selection-chart", "error-chart"].forEach(id => chartObserver.observe($(id)));

/** One-time, idempotent render-sink codemod. Settings and event payloads stay untouched. */
const fs=require('node:fs'),path=require('node:path');
const ts=require(path.resolve(__dirname,'../frontend/node_modules/typescript'));
const root=path.resolve(process.argv[2]||path.join(__dirname,'..'));
const src=path.join(root,'frontend/src');
function files(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?files(path.join(dir,e.name)):e.name.endsWith('.tsx')?[path.join(dir,e.name)]:[])}
const attrs=new Set(['aria-label','aria-valuetext','aria-description','title','placeholder','alt']);
function jsxText(value){const lines=value.split(/\r?\n/).map((line,i,a)=>{let v=line.replace(/\t/g,' ');if(i)v=v.replace(/^ +/,'');if(i<a.length-1)v=v.replace(/ +$/,'');return v;});return lines.filter(Boolean).join(' ')}
let changed=0;
for(const file of files(src)){
 if(file.includes(path.sep+'i18n'+path.sep)||file.endsWith('LanguagePicker.tsx'))continue;
 const text=fs.readFileSync(file,'utf8');if(text.includes('// localized-render-sinks'))continue;
 const source=ts.createSourceFile(file,text,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);const edits=[];
 function replace(node,replacement){edits.push({start:node.getStart(source),end:node.end,text:replacement})}
 function wrap(expression){if(ts.isCallExpression(expression)&&expression.expression.getText(source)==='ui')return;const value=expression.getText(source);replace(expression,`ui(${value})`)}
 function visit(node,raw=false){
   if(ts.isJsxElement(node)){
     const tag=node.openingElement.tagName.getText(source);
     const classes=node.openingElement.attributes.properties.filter(ts.isJsxAttribute).find(a=>a.name.getText(source)==='className')?.initializer?.getText(source)||'';
     raw=raw||tag==='pre'||tag==='code'||classes.includes('log-lines');
   }
   if(raw)return;
   if(ts.isJsxText(node)&&/[\u3400-\u9fff]/.test(node.text)){
     const value=jsxText(node.text);edits.push({start:node.pos,end:node.end,text:`{ui(${JSON.stringify(value)})}`});return;
   }
   if(ts.isJsxExpression(node)&&node.expression){
     const parent=node.parent;
     if(ts.isJsxElement(parent)||ts.isJsxFragment(parent)){
       // The expression may contain further JSX, so use insertion edits and still visit it.
       const expression=node.expression;
       if(!(ts.isCallExpression(expression)&&expression.expression.getText(source)==='ui')){
         edits.push({start:expression.getStart(source),end:expression.getStart(source),text:'ui('});
         edits.push({start:expression.end,end:expression.end,text:')'});
       }
     }
   }
   if(ts.isJsxAttribute(node)&&attrs.has(node.name.getText(source))&&node.initializer){
     const value=node.initializer;
     if(ts.isStringLiteral(value)&&/[\u3400-\u9fff]/.test(value.text))replace(value,`{ui(${JSON.stringify(value.text)})}`);
     else if(ts.isJsxExpression(value)&&value.expression)wrap(value.expression);
   }
   if(ts.isCallExpression(node)&&['confirm','window.confirm','alert','window.alert','prompt','window.prompt'].includes(node.expression.getText(source))&&node.arguments[0])wrap(node.arguments[0]);
   ts.forEachChild(node,child=>visit(child,raw));
 }
 visit(source);
 if(!edits.length)continue;
 let output=text;
 // Insertions at the same boundary retain outer/inner closing order naturally.
 edits.sort((a,b)=>b.start-a.start||b.end-a.end);
 for(const edit of edits)output=output.slice(0,edit.start)+edit.text+output.slice(edit.end);
 let relative=path.relative(path.dirname(file),path.join(src,'i18n/locale')).replaceAll('\\','/');if(!relative.startsWith('.'))relative='./'+relative;
 output=`// localized-render-sinks\nimport {ui} from '${relative}'\n`+output;
 fs.writeFileSync(file,output);changed++;
}
console.log(JSON.stringify({root,localizedComponents:changed}));

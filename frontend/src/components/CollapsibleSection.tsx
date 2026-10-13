import React,{useState} from 'react'

export function CollapsibleSection({title,children,className=''}:{title:React.ReactNode;children:React.ReactNode;className?:string}) {
  const [open,setOpen]=useState(true)
  return <section className={`fold-section ${className}`}><button type="button" className="fold-heading" aria-expanded={open}
    onClick={()=>setOpen(!open)}><span>{title}</span><span aria-hidden="true">{open?'−':'＋'}</span></button>
    <div hidden={!open}>{children}</div></section>
}

export function foldSections(node:React.ReactNode):React.ReactNode {
  return React.Children.map(node,child=>{
    if(!React.isValidElement<{children?:React.ReactNode;className?:string}>(child))return child
    const items=React.Children.toArray(child.props.children)
    const heading=items[0]
    if(child.type==='section'&&React.isValidElement<{className?:string}>(heading)&&
        (heading.type==='h3'||/dlss-card-head|qi-section-title/.test(heading.props.className||''))){
      return <CollapsibleSection key={child.key} className={child.props.className} title={heading}>{foldSections(items.slice(1))}</CollapsibleSection>
    }
    return items.length?React.cloneElement(child,{},foldSections(child.props.children)):child
  })
}

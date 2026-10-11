export function Brand({compact = false}: {compact?: boolean}) {
  return <div className="brand"><img src="./lunarc.svg" alt="Lunarc 光弧标志" width={36} height={36}/>{!compact && <div><strong>Lunarc</strong><small>LIGHT IN MOTION</small></div>}</div>
}

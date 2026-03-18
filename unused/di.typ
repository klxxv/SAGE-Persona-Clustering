// di.typ - Digital Inhumanities Theme Template (Updated)

#let mit-red = rgb("#A31F34")
#let light-gray = rgb("#555555")

#let conf(
  title: "",
  authors: (),
  abstract: [],
  keywords: (),
  body
) = {
  // 页面设置
  set page(
    paper: "a4",
    margin: (x: 2.5cm, y: 3cm),
    header: [
      #set text(8pt, fill: mit-red)
      #smallcaps[Digital Inhumanities] · #datetime.today().display("[year] / [month] / [day]")
      #line(length: 100%, stroke: 1.5pt + mit-red)
      #move(dy: -4pt, line(length: 100%, stroke: 0.5pt + mit-red))
    ],
    footer: [
      #set align(center)
      #set text(8pt, fill: light-gray)
      #line(length: 100%, stroke: 0.5pt + light-gray)
      #context counter(page).display("1")
    ]
  )

  // 字体设置
  let body-font = ("Times New Roman", "SimSun", "STSong")
  let sans-font = ("Arial", "Source Han Sans SC", "Microsoft YaHei", "SimHei")

  
  set text(font: body-font, size: 10.5pt, lang: "zh", region: "cn")

  // 段落设置
  set par(justify: true, first-line-indent: (amount: 2em, all: true), leading: 0.8em)
  
  // 标题样式
  show heading: it => {
    set text(font: sans-font, weight: "bold", fill: mit-red)
    let space-above = if it.level == 1 { 1.8em } else { 1.2em }
    let space-below = if it.level == 1 { 1.2em } else { 0.8em }
    v(space-above, weak: true)
    it
    v(space-below, weak: true)
  }

  // 论文标题
  align(center)[
    #v(1em)
    #block(text(font: sans-font, weight: "bold", size: 22pt, fill: mit-red)[#title])
    #v(0.5em)
    #stack(
      dir: ltr,
      spacing: 1em,
      ..authors.map(a => text(size: 11pt, weight: "medium")[#a])
    )
    #v(1.5em)
  ]

  // 摘要与关键词
  block(inset: (x: 2em))[
    #set text(size: 9pt)
    #set par(first-line-indent: 0pt)
    #text(font: sans-font, weight: "bold")[【摘要】] #abstract
    #v(0.5em)
    #text(font: sans-font, weight: "bold")[【关键词】] #keywords.join("；")
  ]
  
  v(1em)

  // 正文开始
  body
}

// 新的数学小贴士样式：侧栏盒子感
#let math-note(title: "数学小贴士", content) = {
  v(1em)
  block(
    width: 100%,
    stroke: 0.5pt + mit-red,
    radius: 2pt,
    clip: true,
  )[
    #block(
      fill: mit-red,
      width: 100%,
      inset: (x: 0.8em, y: 0.5em),
    )[
      #set text(fill: white, weight: "bold", size: 9pt)
      #title
    ]
    #block(
      fill: mit-red.lighten(95%),
      width: 100%,
      inset: 1em,
    )[
      #set text(size: 9pt)
      #set par(first-line-indent: 0em)
      #content
    ]
  ]
  v(1em)
}

// 参考文献专用样式 (悬挂缩进)
#let bibliography-item(content) = {
  set par(first-line-indent: -1.5em, hanging-indent: 1.5em)
  set text(size: 9pt)
  v(0.5em)
  content
}

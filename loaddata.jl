#load data
#https://github.com/tom880245/kaggle.git

using DataFrames
using CSV
using CairoMakie
import Images
#https://blog.csdn.net/hfy1237/article/details/124663893 繪圖介紹
#畫圖參數
struct PlotConfig
    #axis 常用參數
    #標題
    title::String
    titlealign::Symbol 
    titlecolor::Symbol
    titlefont::Symbol
    titlegap::Float64
    titlesize::Float64

    subtitle::String
    subtitlefont::Symbol
    subtitlecolor::Symbol
    subtitlegap::Int
    subtitlelineheight::Int
    subtitlesize::Float64
    subvisible::Bool

    title_size::Int   #test 
    file_path::String
    basic_path::String
    colors::Vector{Symbol}
    result::String
    xlabel::String
    ylabel::String
    subtitle2::String
    titlecolor2::Symbol
    titlesize3::Int
    ax::Int
    subtitlecolor2::Symbol
    subtitlesize22::Int
    xlabelcolor::Symbol
    xlabelsize::Int
    ylabelcolor::Symbol
    ylabelsize::Int
    
    xgridvisible::Bool
    ygridvisible::Bool
    xgridcolor::Symbol 
    ygridcolor::Symbol
    xgridstyle::Symbol
    ygridstyle::Symbol
    ylabelrotation::Int
    targetcolor::Tuple
    lengendtitle::Array
end
# instantiate   PlotConfig
default_config = PlotConfig(
    "房子\n統計",                #title::String  
    :center,                             #titlealign
    :black,                              #titlecolor
    :blod,                               #titlefont  :italic,:bold ,:bold_italic,:italic,:regular
    4.0,                                 #titlegap
    20.0,                                  #titlesize

    "count/年度",                          #subtitle::String
    :regular   ,                           #subtitlefont::Symbol
    :black     ,                           #subtitlecolor::Symbol
    1         ,                           #subtitlegap
    1         ,                           #subtitlelineheight
    10.0       ,                           #subtitlesize::Float64
    true       ,                           # subvisible::Bool

    25, 
    "1223.txt",                           #file_path
    "basic.txt" ,                          #bascic_path
    [:purple,:darkblue,:blue ,:darkgreen,:green,:yellow], # colors
    "0104.png",                                     #result
    "bachsize",                                       #xlabel::String
    "Time(s)",                                     # ylabel::String
    "10 epochs",              # subtitle::String
    :black,                                     #titlecolor2::Symbol
    28,                                         # titlesize::Int
    1,                                           #ax::Int  基本上要是1
    :darkgray ,                                   #subtitlecolor= :darkgray ,
    18 ,                                      #subtitlesize= 18 ,
    :darkcyan,                                       #xlabelcolor=:darkcyan,
    20,                                       #xlabelsize=20,
    :darkcyan,                                       #ylabelcolor=:darkcyan,
    20,                                      #ylabelsize=20,
    true,                                     #xgridvisible = true, 
    true,                                    #ygridvisible = true,
    :gray,                                     #xgridcolor = :gray, 
    :gray,                                     #ygridcolor = :gray, 
    :dash,                                     #xgridstyle = :dash, 
    :dash,                                    #ygridstyle = :dash,
    0,                                   #ylabelrotation=0,
    (:red,:darkred,:orange,:darkorange,:yellow,:drakyellow) ,                # targetcolor   1
    ["10 epoch", "Full epoch"]              #lengendtitle
    
)

joinpath(@__DIR__,"air_pollution.csv")
# 指定CSV檔案的路
path=@__DIR__
file_path = joinpath(@__DIR__,"air_pollution.csv")

# 使用CSV.File()函數讀取CSV檔案並轉換為DataFrame
df = DataFrame(CSV.File(file_path))

#可畫圖片 1. 各國家城市數量
#2.各國家平均PM2.5 對年度
#3.預測2024 各城市的PM2.5  和各國家的PM2.5

#1. 各國家城市數量

# 使用groupby和combine函數進行組合和計數
result = combine(groupby(df, :country), nrow)
sorted_result = sort(result, :nrow, rev=true)

total_nrow = sum(sorted_result.nrow)
#篩選出>100的 依序排序
filtered_result = filter(row -> row.nrow > 100, result)

# 使用eachrow迭代DataFrame的行，並根據條件收集索引
filtered_indexes = Int[]
for (index, row) in enumerate(eachrow(result))
    if row.nrow > 100
        push!(filtered_indexes, index)
    end
end

df_100=filter(:country => x -> x in collect(filtered_result.country)  ,df)

#畫圖
f=Figure()
ax=?Axis(f[1,1],title=default_config.title,titlecolor=Images.RGB(0.2, 0.1, 0.1),titlefont=:bold_italic,titlegap=8.0,titlesize=20,titlelineheight=1

,subtitle=default_config.subtitle,Aspect= AxisAspect(1))


lines!(ax, parse.(Int,names(df_100[1,3:end])),collect(df_100[1,3:end]))
#test
# collect(df_100[1,3:end])

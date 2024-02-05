using Gtk
# 创建一个名为 "A new window" 的 Gtk 窗口
# win = GtkWindow("A new window")

# # 创建一个 GtkGrid 用于容纳小部件
# g = GtkGrid()

# # 创建一个 GtkEntry，用于输入文本
# a = GtkEntry()
# set_gtk_property!(a, :text, "This is Gtk!")  # 设置 GtkEntry 的文本内容

# # 创建一个 GtkCheckButton，用于创建复选框
# b = GtkCheckButton("Check me!")

# # 创建一个 GtkScale，用于创建一个滑动条
# c = GtkScale(false, 0:10)  # 创建一个不显示刻度标签的滑动条，范围从 0 到 10

# # 将这些图形元素放置到 Grid 中
# g[1,1] = a    # 使用 Cartesian 坐标将 GtkEntry 放在第一行第一列
# g[2,1] = b    # 将 GtkCheckButton 放在第二行第一列
# g[1:2,2] = c  # 跨越两列放置 GtkScale
# set_gtk_property!(g, :column_homogeneous, true)  # 设置每個元素 列的尺寸均匀
# set_gtk_property!(g, :column_spacing, 15)  # 在列之间引入 15 像素的间隔

# # 将 Grid 添加到窗口中
# push!(win, g)

# # 显示窗口
# showall(win)


#Signals and Callbacks
# b = GtkButton("Press me")
# win = GtkWindow(b, "Callbacks")
# showall(win)

# function button_clicked_callback(widget)
#     println(widget, " was clicked!")
# end
# id = signal_connect(b, "button-press-event") do widget, event
#      println("You pressed button ", event.button)
#  end
b = GtkBuilder(filename="myapp.glade")